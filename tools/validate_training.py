"""Bounded, reproducible CPU training and matched-seed greedy evaluation."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import platform
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from ai.arena import Arena
from ai.checkpoints import CheckpointStore
from ai.dqn import Trainer
from ai.training_config import TrainingConfig


def evaluate(network,config,seeds):
    results=[]
    for seed in seeds:
        arena=Arena(copy.deepcopy(network),config,seed)
        while not arena.environment.done:
            arena.step()
        results.append(dict(seed=seed,score=arena.environment.game.score,steps=arena.environment.steps,reason=arena.last_reason))
    return dict(average_score=sum(row['score'] for row in results)/len(results),
                terminations=dict(Counter(row['reason'] for row in results)),games=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,default=300)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--output',type=Path,default=Path('.artifacts/v14-validation'))
    args=parser.parse_args()
    torch.set_num_threads(1)
    config=TrainingConfig(episodes=args.episodes,seed=args.seed,replay_capacity=10000,
        epsilon_decay_steps=10000,target_sync=250,episode_cap=1000,checkpoint_interval=100)
    trainer=Trainer(config)
    seeds=list(range(20))
    before=evaluate(trainer.online,config,seeds)
    store=CheckpointStore(args.output/'checkpoints')
    args.output.mkdir(parents=True,exist_ok=True)
    from uuid import uuid4
    run=uuid4().hex
    store.save(run,'Validation DQN',trainer)
    start=time.perf_counter()
    while trainer.completed<config.episodes:
        trainer.train_step()
    seconds=time.perf_counter()-start
    trainer.elapsed=seconds
    checkpoint=store.save(run,'Validation DQN',trainer)
    reloaded,metadata=store.resume(checkpoint)
    assert all(torch.equal(a,b) for a,b in zip(trainer.online.parameters(),reloaded.online.parameters()))
    after=evaluate(reloaded.online,config,seeds)
    report=dict(config=config.to_dict(),architecture=trainer.online.config.to_dict(),evaluation_seeds=seeds,
        device='cpu',torch=str(torch.__version__),python=platform.python_version(),
        training_seconds=seconds,episodes_per_second=trainer.completed/seconds,
        environment_steps_per_second=trainer.total_steps/seconds,environment_steps=trainer.total_steps,
        completed_episodes=trainer.completed,checkpoint=str(checkpoint),checkpoint_round_trip_exact=True,
        before=before,after=after,metrics=trainer.metrics(),
        history=dict(scores=list(trainer.scores),loss=list(trainer.losses),epsilon=list(trainer.epsilons)))
    (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key not in ('history','config','architecture')},indent=2))


if __name__=='__main__':
    main()
