"""Short paired kernel check against a user-supplied historical source snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import time
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def historical(root):
    environment=types.ModuleType('snake_historical_environment')
    sys.modules[environment.__name__]=environment
    exec(compile((root/'ai/environment.py').read_text(encoding='utf-8'),str(root/'ai/environment.py'),'exec'),environment.__dict__)
    training=types.ModuleType('snake_historical_dqn')
    sys.modules[training.__name__]=training
    exec(compile((root/'ai/dqn.py').read_text(encoding='utf-8'),str(root/'ai/dqn.py'),'exec'),training.__dict__)
    training.Episode=environment.Episode
    return training.Trainer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=4096)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--output',type=Path,default=Path('.artifacts/v152-legacy-compat.json'))
    args=parser.parse_args()
    if args.steps<64 or not 1<=args.repeats<=3:parser.error('Use at least 64 steps and 1..3 repeats')
    import torch
    torch.set_num_threads(1)
    from ai.dqn import Trainer
    from ai.training_config import TrainingConfig
    old=historical(args.baseline_root)
    config=TrainingConfig(episodes=1000000,seed=7)
    results=[]
    for repeat in range(args.repeats):
        trial={}
        order=(('baseline',old),('current',Trainer)) if repeat%2==0 else (('current',Trainer),('baseline',old))
        for name,constructor in order:
            trainer=constructor(config)
            for _ in range(64):trainer.train_step()
            digest=hashlib.sha256()
            started=time.perf_counter()
            for _ in range(args.steps):digest.update(repr(trainer.train_step()).encode('utf-8'))
            duration=time.perf_counter()-started
            weights=hashlib.sha256()
            for parameter in trainer.online.parameters():weights.update(bytes(parameter.detach().view(torch.uint8).flatten().tolist()))
            trial[name]=dict(seconds=duration,steps_per_second=args.steps/duration,transitions=digest.hexdigest(),weights=weights.hexdigest())
        trial['identical']=trial['baseline']['transitions']==trial['current']['transitions'] and trial['baseline']['weights']==trial['current']['weights']
        results.append(trial)
    report=dict(steps_per_trial=args.steps,repeats=args.repeats,trials=results,
        baseline_median_steps_per_second=median(r['baseline']['steps_per_second'] for r in results),
        current_median_steps_per_second=median(r['current']['steps_per_second'] for r in results),
        all_identical=all(r['identical'] for r in results))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
