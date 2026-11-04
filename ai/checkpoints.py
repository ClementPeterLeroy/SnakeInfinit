"""Shared trusted-local checkpoint library, including complete boundary resume state."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
from uuid import uuid4
import zipfile
import torch
from ai.persistence import ModelStore
from ai.training_config import TrainingConfig
from ai.config import NetworkConfig
from ai.environment import Transition
from ai.dqn import Trainer
from services.userdata import user_data_dir
from services.checkpoint_catalog import checkpoint_roots
from ai.model_registry import ModelSpec,identity,parse_model_config

REASONS = (None, 'wall_collision','self_collision','stagnation','episode_step_limit','board_completion')


def validated_weights(network, state):
    expected = network.state_dict()
    if not isinstance(state,dict) or set(state) != set(expected):
        raise ValueError("Incompatible checkpoint network")
    for key, value in state.items():
        if not isinstance(value,torch.Tensor) or value.dtype != torch.float32 or value.shape != expected[key].shape or not torch.isfinite(value).all():
            raise ValueError("Invalid checkpoint tensor: " + key)
    return state


class CheckpointStore:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else user_data_dir() / 'checkpoints'

    def entries(self):
        roots=checkpoint_roots(self.root)
        if len(roots)>1:
            result=[]
            for root in roots:
                result.extend(CheckpointStore(root)._local_entries())
            return sorted(result,key=lambda x:(x['run'],x['checkpoint']))
        return self._local_entries()

    def _local_entries(self):
        if not self.root.exists():
            return []
        result = []
        for run in sorted(self.root.iterdir()):
            if not run.is_dir() or run.is_symlink():
                continue
            for name in ModelStore(run).names():
                folder = ModelStore(run).path(name)
                if (folder / 'training.json').is_file():
                    try:
                        data = self.metadata(folder)
                        if isinstance(data,dict):
                            result.append(dict(run=run.name, checkpoint=name, name=str(data.get('model_name',run.name))[:40],
                                episode=data.get('episode',0), path=folder))
                    except (ValueError,OSError):
                        continue
        return sorted(result,key=lambda x:(x['run'], x['checkpoint']))

    def save(self, run, name, trainer, checkpoint=None):
        if not isinstance(name,str) or not 1<=len(name)<=40:
            raise ValueError('Model name must contain 1..40 characters')
        if not trainer.at_boundary:
            raise ValueError("Resume checkpoints require an episode boundary")
        if hasattr(trainer,'validate_recovery'):trainer.validate_recovery()
        run_path = ModelStore(self.root).path(run)
        checkpoint = checkpoint or f'episode-{trainer.completed:07d}-{uuid4().hex[:8]}'
        models = ModelStore(run_path)
        # Existing destinations are never overwritten, including incomplete saves.
        folder = models.save(checkpoint, trainer.online,reinforcement_learning=True)
        replay = trainer.replay.items
        observations=trainer.codec.pack([t.observation for t in replay]) if trainer.codec else torch.tensor([t.observation for t in replay],dtype=torch.uint8).reshape(-1,11)
        next_observations=trainer.codec.pack([t.next_observation for t in replay]) if trainer.codec else torch.tensor([t.next_observation for t in replay],dtype=torch.uint8).reshape(-1,11)
        state = dict(target=trainer.target.state_dict(), optimizer=trainer.optimizer.state_dict(),
            observations=observations,
            next_observations=next_observations,
            actions=torch.tensor([t.action for t in replay],dtype=torch.int64),
            rewards=torch.tensor([t.reward for t in replay],dtype=torch.float64),
            flags=torch.tensor([(t.terminated,t.truncated) for t in replay],dtype=torch.bool).reshape(-1,2),
            reasons=[t.reason for t in replay], replay_position=trainer.replay.position,
            replay_rng=trainer.replay.rng.getstate(), action_rng=trainer.action_rng.getstate(),
            environment_rng=trainer.environment_rng.getstate(), torch_rng=trainer.torch_rng_state,
            counters={k:getattr(trainer,k) for k in ('completed','total_steps','apples','collisions','stagnations','best')},
            history={k:list(getattr(trainer,k)) for k in ('scores','lengths','rewards','losses','epsilons')},
            loss=trainer.loss, gradient_norm=trainer.gradient_norm, elapsed=trainer.elapsed,
            cumulative_reward=trainer.cumulative_reward,last_reason=trainer.last_reason,
            optimizer_updates=trainer.optimizer_updates)
        if hasattr(trainer,'snapshot_state'):state['parallel']=trainer.snapshot_state()
        resume = folder/'training.pt'
        torch.save(state,resume)
        metadata = dict(format_version=2 if trainer.codec else 1, kind='resumable-boundary', model_identifier=run, model_name=name,
            architecture=trainer.online.config.to_dict(), episode=trainer.completed, environment_steps=trainer.total_steps,
            epsilon=trainer.epsilon, optimizer_updates=trainer.optimizer_updates,hyperparameters=trainer.config.to_dict(),
            metrics=trainer.metrics(), created_utc=datetime.now(timezone.utc).isoformat(),
            torch_version=str(torch.__version__), sha256=hashlib.sha256(resume.read_bytes()).hexdigest(),
            truncation_convention='finite episodic return; no bootstrap for termination or truncation')
        metadata.update(identity(trainer.online.config))
        if getattr(trainer,'generation_id',None):metadata['generation_id']=trainer.generation_id
        if getattr(trainer,'generation_number',None):metadata['generation_number']=trainer.generation_number
        if getattr(trainer,'training_interface',None):metadata['training_interface']=trainer.training_interface
        if hasattr(trainer,'execution'):metadata['parallel_execution']=trainer.execution.to_dict()
        temporary = folder/'training.json.tmp'
        temporary.write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(folder/'training.json')  # Library visibility commits last.
        return folder

    def _folder(self, folder):
        folder = Path(folder)
        roots=checkpoint_roots(self.root)
        root=next((root for root in roots if folder.resolve().parent.parent==root.resolve()),self.root)
        run = ModelStore(root).path(folder.parent.name)
        expected = ModelStore(run).path(folder.name)
        if folder.resolve() != expected.resolve():
            raise ValueError("Checkpoint must belong to the local library")
        return expected

    def load_model(self, folder):
        folder = self._folder(folder)
        model, _ = ModelStore(folder.parent).load(folder.name)
        data = self.metadata(folder)
        if model.config != parse_model_config(data):
            raise ValueError("Checkpoint architecture metadata mismatch")
        return model, data

    def metadata(self, folder):
        folder = self._folder(folder)
        path = folder/'training.json'
        if path.is_symlink() or path.stat().st_size > 200000:
            raise ValueError("Invalid checkpoint metadata")
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data,dict) or data.get('format_version') not in (1,2) or data.get('kind') not in ('resumable-boundary','model-snapshot'):
            raise ValueError("Unsupported checkpoint format")
        config=TrainingConfig.from_dict(data.get('hyperparameters'))
        parse_model_config(data)
        if (data.get('model_identifier')!=folder.parent.name or not isinstance(data.get('model_name'),str)
                or not 1<=len(data['model_name'])<=40
                or type(data.get('episode')) is not int or not 0<=data['episode']<=config.episodes
                or type(data.get('environment_steps')) is not int or data['environment_steps']<data['episode']
                or type(data.get('epsilon')) not in (float,int) or not math.isfinite(data['epsilon']) or not 0<=data['epsilon']<=1
                or not isinstance(data.get('sha256'),str) or re.fullmatch('[0-9a-f]{64}',data['sha256']) is None):
            raise ValueError('Invalid checkpoint identity or counters')
        return data

    def resume(self, folder):
        try:
            model, metadata = self.load_model(folder)
            if metadata['kind']!='resumable-boundary':
                raise ValueError('Model-only snapshot cannot resume training; choose a recovery or final checkpoint')
            folder = self._folder(folder)
            path = folder/'training.pt'
            if path.is_symlink() or path.stat().st_size > 256_000_000:
                raise ValueError("Checkpoint state is too large or not local")
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['sha256']:
                raise ValueError("Checkpoint checksum mismatch")
            with zipfile.ZipFile(path) as archive:
                if sum(item.file_size for item in archive.infolist()) > 512_000_000:
                    raise ValueError("Expanded checkpoint is too large")
            state = torch.load(path,map_location='cpu',weights_only=True)
            trainer = Trainer(TrainingConfig.from_dict(metadata['hyperparameters']),network=model)
            trainer.target.load_state_dict(validated_weights(trainer.target,state['target']))
            n = len(state['actions'])
            specs = dict(actions=((n,),torch.int64),rewards=((n,),torch.float64),flags=((n,2),torch.bool))
            if trainer.codec:
                observations=trainer.codec.unpack(state['observations'],n)
                next_observations=trainer.codec.unpack(state['next_observations'],n)
            else:
                specs.update(observations=((n,11),torch.uint8),next_observations=((n,11),torch.uint8))
            if n > trainer.config.replay_capacity or len(state['reasons']) != n:
                raise ValueError("Invalid replay capacity")
            for key,(shape,dtype) in specs.items():
                value=state[key]
                if not isinstance(value,torch.Tensor) or value.shape != shape or value.dtype != dtype or not torch.isfinite(value).all():
                    raise ValueError("Invalid replay tensor: " + key)
            if ((not trainer.codec and ((state['observations'] > 1).any() or (state['next_observations'] > 1).any()))
                    or (state['actions'] < 0).any() or (state['actions'] > 2).any()
                    or any(reason not in REASONS for reason in state['reasons'])):
                raise ValueError("Invalid replay values")
            for i in range(n):
                trainer.replay.append(Transition(observations[i] if trainer.codec else tuple(state['observations'][i].tolist()),int(state['actions'][i]),
                    float(state['rewards'][i]),next_observations[i] if trainer.codec else tuple(state['next_observations'][i].tolist()),
                    bool(state['flags'][i,0]),bool(state['flags'][i,1]),state['reasons'][i]))
            position = state['replay_position']
            if type(position) is not int or not 0 <= position < trainer.config.replay_capacity:
                raise ValueError("Invalid replay cursor")
            trainer.replay.position = position
            for key in ('completed','total_steps','apples','collisions','stagnations','best'):
                value = state['counters'][key]
                if type(value) is not int or value < 0:
                    raise ValueError("Invalid training counter")
                setattr(trainer,key,value)
            if trainer.completed > trainer.config.episodes or trainer.total_steps < trainer.completed:
                raise ValueError("Inconsistent episode counters")
            for key in ('scores','lengths','rewards','losses','epsilons'):
                values=state['history'][key]
                if not isinstance(values,list) or len(values)>getattr(trainer,key).maxlen or any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
                    raise ValueError("Invalid training history")
                getattr(trainer,key).extend(values)
            for key in ('loss','gradient_norm','elapsed','cumulative_reward'):
                value=state[key]
                if value is not None and (type(value) not in (float,int) or not math.isfinite(value)):
                    raise ValueError("Invalid training metric")
                setattr(trainer,key,value)
            trainer.last_reason=state['last_reason']
            if trainer.last_reason not in REASONS or trainer.elapsed is None or trainer.elapsed<0:
                raise ValueError('Invalid training reason or elapsed time')
            trainer.replay.rng.setstate(state['replay_rng'])
            trainer.action_rng.setstate(state['action_rng'])
            trainer.environment_rng.setstate(state['environment_rng'])
            generator=torch.Generator()
            generator.set_state(state['torch_rng'])
            trainer.torch_rng_state=generator.get_state()
            optimizer=state['optimizer']
            baseline=trainer.optimizer.state_dict()['param_groups']
            if optimizer['param_groups'] != baseline:
                raise ValueError("Invalid optimizer configuration")
            expected_ids=set(baseline[0]['params'])
            if set(optimizer['state']) not in (set(),expected_ids):
                raise ValueError('Invalid optimizer parameter identities')
            for saved in optimizer['state'].values():
                if set(saved)!= {'step','exp_avg','exp_avg_sq'}:
                    raise ValueError('Invalid Adam state fields')
            trainer.optimizer.load_state_dict(optimizer)
            for parameter, saved in trainer.optimizer.state.items():
                for key,value in saved.items():
                    if (not isinstance(value,torch.Tensor) or not torch.isfinite(value).all()
                            or key=='step' and (value.numel()!=1 or value.item()<0)
                            or key!='step' and (value.shape!=parameter.shape or value.dtype!=parameter.dtype)):
                        raise ValueError("Invalid Adam tensor")
            if metadata['episode'] != trainer.completed or metadata['environment_steps'] != trainer.total_steps or abs(metadata['epsilon']-trainer.epsilon)>1e-12:
                raise ValueError("Inconsistent checkpoint metadata")
            updates=state.get('optimizer_updates',max(0,trainer.total_steps-trainer.config.batch_size+1))
            if type(updates) is not int or not 0<=updates<=trainer.total_steps:raise ValueError('Invalid optimizer update counter')
            if metadata.get('optimizer_updates',updates)!=updates:raise ValueError('Optimizer update metadata mismatch')
            trainer.optimizer_updates=updates
            if 'parallel' in state or 'parallel_execution' in metadata:
                from ai.frontier_config import FrontierConfig
                from ai.parallel import ParallelTrainer
                execution=FrontierConfig.from_dict(metadata['parallel_execution'])
                if execution.to_dict()!=FrontierConfig.from_dict(state['parallel']['execution']).to_dict():raise ValueError('Parallel checkpoint policy mismatch')
                parallel=ParallelTrainer(trainer.config,execution,learner=trainer)
                try:parallel.restore_state(state['parallel'])
                except BaseException:parallel.close();raise
                trainer=parallel
            return trainer, metadata
        except (OSError,RuntimeError,EOFError,TypeError,KeyError,IndexError,pickle.UnpicklingError,zipfile.BadZipFile) as error:
            raise ValueError('Cannot resume checkpoint: '+str(error)) from error
