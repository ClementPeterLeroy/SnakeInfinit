"""Cancelable frozen same-seed evaluations; no training model is shared."""
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
from uuid import uuid4
from ai.analysis_stats import EpisodeRecord


def evaluate_checkpoints(paths,seeds,output_root,cancel=None,progress=None,configuration=None):
    from ai.arena import Arena
    from ai.checkpoints import CheckpointStore
    from ai.training_config import TrainingConfig
    if not 1<=len(paths)<=2 or not seeds or len(seeds)>10000 or any(type(s) is not int or not 0<=s<2**63 for s in seeds):
        raise ValueError('Evaluation requires 1–2 checkpoints and 1–10,000 valid seeds')
    paths=[Path(path) for path in paths]
    loaded=[]
    for path in paths:
        store=CheckpointStore(path.parent.parent)
        model,metadata=store.load_model(path)
        loaded.append((model,metadata,path))
    config=TrainingConfig.from_dict(configuration or loaded[0][1]['hyperparameters'])
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
            architecture=model.config.sizes,model_identity=identity(model.config),run_id=metadata['model_identifier'],records=[asdict(row) for row in records]))
    payload=dict(schema_version=1,identifier=identifier,dataset_type='evaluation',policy='frozen greedy; epsilon=0; no optimizer',
        seeds=list(seeds),configuration=config.to_dict(),models=models,created_utc=datetime.now(timezone.utc).isoformat())
    output_root=Path(output_root)
    output_root.mkdir(parents=True,exist_ok=True)
    path=output_root/(identifier+'.json')
    with path.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,indent=2,allow_nan=False)
    return path
