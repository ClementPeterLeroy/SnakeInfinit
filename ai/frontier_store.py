"""Isolated experiment artifacts; checkpoint visibility commits metadata last."""
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4
import torch
from ai.checkpoints import CheckpointStore,validated_weights
from ai.config import NetworkConfig
from ai.actions import ACTION_NAMES
from ai.observations import FEATURE_NAMES
from ai.experiments import BASELINE_ENCODER
from ai.network import SnakeNetwork
from ai.persistence import ModelStore
from core.config import VERSION
from ai.model_registry import ModelSpec,identity,parse_model_config,build_model,LINEUP
from ai.encoders import get_encoder


def atomic_json(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    try:
        with temporary.open('x',encoding='utf-8') as stream:
            json.dump(data,stream,indent=2,ensure_ascii=False,allow_nan=False)
            stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


class FrontierStore:
    def __init__(self,output,profile,config,execution,name,identifier=None,resumed_from=None):
        if not isinstance(name,str) or not 1<=len(name.strip())<=40:raise ValueError('Model name must contain 1..40 characters')
        self.identifier=identifier or uuid4().hex
        if not self.identifier.isalnum() or len(self.identifier)>40:raise ValueError('Invalid experiment identifier')
        self.root=Path(output).resolve()/self.identifier
        self.root.mkdir(parents=True,exist_ok=False)
        self.lock=self.root/'.writer.lock'
        self.fd=os.open(self.lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        self.name=name
        self.started=time.perf_counter()
        self.checkpoints=CheckpointStore(self.root/'checkpoints')
        self.recoveries=[]
        self.artifacts=[]
        self.metadata=dict(schema_version=1,application_version=VERSION,experiment_id=self.identifier,
            name=name,encoder_id=BASELINE_ENCODER,input_count=11,training=config.to_dict(),execution=execution.to_dict(),
            training_seed=config.seed,evaluation_seeds=list(range(25)),environment_step_budget=config.episodes*config.episode_cap,
            resumed_from=str(resumed_from) if resumed_from else None,created_utc=datetime.now(timezone.utc).isoformat(),
            history_root=str(self.root/'history'),checkpoint_root=str(self.checkpoints.root),
            resume_strategy='Completed-episode boundary; replay, optimizer and RNG retained. Partial episodes are not serialized.')
        try:
            atomic_json(self.root/'experiment.json',self.metadata)
            atomic_json(Path(profile)/'frontier_index'/(self.identifier+'.json'),
                dict(schema_version=1,checkpoint_root=str(self.checkpoints.root),experiment_root=str(self.root)))
        except Exception:
            self.close()
            raise

    def details(self,trainer):
        info=identity(trainer.online.config)
        return dict(experiment_id=self.identifier,observation_encoder=info['encoder_id'],
            history_root=str(self.root/'history'),history_run=self.identifier,
            architecture=trainer.online.config.to_dict(),resume_strategy=self.metadata['resume_strategy'],
            collection_configuration=self.metadata['execution'],optimizer_updates=trainer.optimizer_updates,
            training_elapsed_seconds=trainer.elapsed,session_wall_seconds=time.perf_counter()-self.started,
            target_sync_basis='global environment transitions',epsilon_basis='global environment transitions',**info)

    def snapshot(self,trainer):
        """Light model-only snapshot, plus compatible Arena/Analysis library metadata."""
        folder=ModelStore(self.checkpoints.root/self.identifier).save(
            f'episode-{trainer.completed:07d}-{uuid4().hex[:8]}',trainer.online,reinforcement_learning=True)
        network=json.loads((folder/'network.json').read_text(encoding='utf-8'))
        metadata=dict(format_version=2 if trainer.codec else 1,kind='model-snapshot',model_identifier=self.identifier,model_name=self.name,
            episode=trainer.completed,environment_steps=trainer.total_steps,epsilon=trainer.epsilon,
            hyperparameters=trainer.config.to_dict(),metrics=trainer.metrics(),created_utc=datetime.now(timezone.utc).isoformat(),
            sha256=network['sha256'],**self.details(trainer))
        atomic_json(folder/'training.json',metadata)
        model=self.publish_model(folder,trainer.completed)
        self.checkpoints.load_model(folder)
        self.artifacts.append(dict(kind='model',episode=trainer.completed,path=str(model),library=str(folder)))
        self.save_manifest()
        return folder,model

    def publish_model(self,folder,episode):
        metadata=json.loads((folder/'network.json').read_text(encoding='utf-8'))
        checkpoint_metadata=json.loads((folder/'training.json').read_text(encoding='utf-8'))
        root=self.root/'models'
        root.mkdir(exist_ok=True)
        # Every snapshot is immutable, including repeated saves at the same episode.
        prefix=LINEUP[metadata.get('model_configuration_id','A')][4]
        basename=f'{prefix}_ep{episode:06d}_model'
        path=root/(basename+'.pt')
        if path.exists():path=root/(basename+'_'+uuid4().hex[:8]+'.pt')
        with path.open('xb') as stream:
            stream.write((folder/metadata['weights_file']).read_bytes());stream.flush();os.fsync(stream.fileno())
        atomic_json(path.with_suffix('.json'),{**metadata,'experiment_id':self.identifier,'model_name':self.name,
            'observation_encoder':metadata.get('encoder_id',BASELINE_ENCODER),'episode':episode,
            'environment_steps':checkpoint_metadata['environment_steps'],
            'optimizer_updates':checkpoint_metadata.get('optimizer_updates'),
            'training_elapsed_seconds':checkpoint_metadata.get('training_elapsed_seconds'),
            'session_wall_seconds':checkpoint_metadata.get('session_wall_seconds'),
            'collection_configuration':checkpoint_metadata.get('collection_configuration',self.metadata['execution']),
            'weights_file':path.name})
        load_published_model(path)  # Verify checksum, shapes and actual reload before reporting success.
        return path

    def save(self,trainer,kind='recovery',slots=2):
        folder=self.checkpoints.save(self.identifier,self.name,trainer,
            checkpoint=f'{kind}-{trainer.completed:07d}-{uuid4().hex[:8]}')
        metadata=json.loads((folder/'training.json').read_text(encoding='utf-8'))
        atomic_json(folder/'training.json',{**metadata,**self.details(trainer)})
        self.checkpoints.load_model(folder)
        # Descriptive resume-state filename accompanies canonical network/config files.
        descriptive=folder/f"{LINEUP[metadata['model_configuration_id']][4]}_ep{trainer.completed:06d}_checkpoint.pt"
        try:os.link(folder/'training.pt',descriptive)
        except OSError:shutil.copyfile(folder/'training.pt',descriptive)
        if kind=='recovery':
            atomic_json(self.root/'recovery.json',dict(schema_version=1,path=str(folder),episode=trainer.completed))
            self.recoveries.append(folder)
            while len(self.recoveries)>slots:
                old=self.recoveries.pop(0)
                if old.resolve().parent!=(self.checkpoints.root/self.identifier).resolve() or not old.name.startswith('recovery-'):
                    raise ValueError('Recovery cleanup target is outside this experiment')
                shutil.rmtree(old)
        model=self.publish_model(folder,trainer.completed) if kind=='final' else None
        self.artifacts.append(dict(kind=kind,episode=trainer.completed,path=str(folder),model=str(model) if model else None))
        self.save_manifest()
        return folder,model

    def save_manifest(self):
        atomic_json(self.root/'artifacts.json',dict(schema_version=1,artifacts=self.artifacts))

    def close(self):
        if self.fd is not None:
            os.close(self.fd);self.fd=None
            self.lock.unlink(missing_ok=True)


def load_published_model(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>8000000:raise ValueError('Invalid published model')
    metadata_path=path.with_suffix('.json')
    if metadata_path.is_symlink() or metadata_path.stat().st_size>64000:raise ValueError('Invalid model metadata')
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('weights_file')!=path.name:raise ValueError('Invalid model metadata')
    config=parse_model_config(metadata)
    features=list(get_encoder(config.encoder_id).ordering) if isinstance(config,ModelSpec) else list(FEATURE_NAMES)
    if metadata.get('features')!=features or metadata.get('actions')!=list(ACTION_NAMES):raise ValueError('Incompatible observation/action metadata')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['sha256']:raise ValueError('Model checksum mismatch')
    model=build_model(config) if isinstance(config,ModelSpec) else SnakeNetwork(config)
    model.load_state_dict(validated_weights(model,torch.load(path,map_location='cpu',weights_only=True)))
    return model.eval(),metadata
