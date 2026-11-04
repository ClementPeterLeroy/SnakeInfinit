"""Cancelable frozen same-seed evaluations; no training model is shared."""
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
from uuid import uuid4
from ai.analysis_stats import EpisodeRecord


def evaluate_checkpoints(paths,seeds,output_root,cancel=None,progress=None,configuration=None,library=None,checkpoint_ids=None):
    tokens=[];loaded=[]
    try:
        for index,source in enumerate(paths):
            path=Path(source)
            if library:
                cp=library.checkpoint(checkpoint_ids[index]) if checkpoint_ids and checkpoint_ids[index] else library.find_checkpoint(path)
                if not cp:raise ValueError('Selected checkpoint is not indexed in the Model Library')
                tokens.append(library.acquire(cp['generation_id'],'Frozen evaluation',cp['id']))
                model,data=library.load(cp['id'])
            elif path.is_file():
                from ai.frontier_store import load_published_model
                model,data=load_published_model(path)
                data={**data,'model_identifier':data.get('experiment_id',data.get('generation_id',path.stem))}
            elif (path/'training.json').is_file():
                from ai.checkpoints import CheckpointStore
                model,data=CheckpointStore(path.parent.parent).load_model(path)
            else:
                from ai.persistence import ModelStore
                model,data=ModelStore(path.parent).load(path.name)
                data={**data,'model_identifier':data.get('generation_id',path.name),'model_name':path.name,'episode':0}
            loaded.append((model,data,path))
        result=_evaluate_loaded(loaded,seeds,output_root,cancel,progress,configuration)
        if library:library._index_references()
        return result
    finally:
        if library:
            for token in tokens:library.release(token)


def _evaluate_loaded(loaded,seeds,output_root,cancel=None,progress=None,configuration=None):
    from ai.arena import Arena
    from ai.checkpoints import CheckpointStore
    from ai.training_config import TrainingConfig
    if not 1<=len(loaded)<=2 or not seeds or len(seeds)>10000 or any(type(s) is not int or not 0<=s<2**63 for s in seeds):
        raise ValueError('Evaluation requires 1–2 checkpoints and 1–10,000 valid seeds')
    config=TrainingConfig.from_dict(configuration or loaded[0][1].get('hyperparameters') or TrainingConfig().to_dict())
    identifier=uuid4().hex
    models=[]
    for model_index,(model,metadata,path) in enumerate(loaded):
        records=[]
        cumulative=0
        arena=Arena(model,config,seeds[0])
        for index,seed in enumerate(seeds):
            arena.reset(seed)
            while not arena.environment.done:
                if cancel and cancel.is_set():
                    raise InterruptedError('Evaluation canceled; no partial result published')
                arena.step()
            env=arena.environment
            cumulative+=env.steps
            records.append(EpisodeRecord(metadata['model_identifier'],index+1,env.game.score,env.steps,
                env.total_reward,env.reason,cumulative,0.,None,env.game.apples_eaten,seed))
            if progress:progress(model_index,index+1,len(seeds))
        from ai.model_registry import identity
        models.append(dict(name=metadata['model_name'],checkpoint=str(path.resolve()),training_episode=metadata['episode'],
            generation_id=metadata.get('generation_id'),library_checkpoint_id=metadata.get('library_checkpoint_id'),
            architecture=model.config.sizes,model_identity=identity(model.config),run_id=metadata['model_identifier'],records=[asdict(row) for row in records]))
    payload=dict(schema_version=1,identifier=identifier,dataset_type='evaluation',policy='frozen greedy; epsilon=0; no optimizer',
        seeds=list(seeds),configuration=config.to_dict(),models=models,created_utc=datetime.now(timezone.utc).isoformat())
    output_root=Path(output_root)
    output_root.mkdir(parents=True,exist_ok=True)
    path=output_root/(identifier+'.json')
    with path.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,indent=2,allow_nan=False)
    return path
