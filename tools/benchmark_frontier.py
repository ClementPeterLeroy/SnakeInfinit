"""Matched real-worker benchmark. Uses isolated profiles; never production saves."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def run(mode,steps,output,profiling=False):
    import torch
    torch.set_num_threads(1)
    from ai.dqn import Trainer
    from ai.training_config import TrainingConfig
    from ai.profiling import process_usage
    from ai.frontier_config import FrontierConfig
    from ai.checkpoints import CheckpointStore
    from ai.frontier import FrontierWorker
    from ai.worker import TrainingWorker
    import ai.frontier as frontier_module
    import ai.worker as legacy_module
    output=Path(output)
    profile=output/'profile'
    capture={}
    trajectory=hashlib.sha256()
    def weights(model):
        digest=hashlib.sha256()
        for parameter in model.parameters():digest.update(bytes(parameter.detach().contiguous().view(torch.uint8).flatten().tolist()))
        return digest.hexdigest()
    class CapturedTrainer(Trainer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            capture['trainer']=self
            capture['initial_weights']=weights(self.online)
        def train_step(self):
            if 'first_step' not in capture:capture['first_step']=time.perf_counter()
            transition=super().train_step()
            trajectory.update(repr(transition).encode('utf-8'))
            if self.total_steps%25000==0:print(f'{mode}: {self.total_steps:,} environment steps',flush=True)
            if self.total_steps>=steps:worker.command('pause')
            return transition
    config=TrainingConfig(episodes=1000000,seed=7)
    if mode=='legacy':
        legacy_module.Trainer=CapturedTrainer
        worker=TrainingWorker(CheckpointStore(profile/'checkpoints'))
        started=time.perf_counter()
        worker.start(config)
        worker.command('view',('headless',1))
    else:
        frontier_module.Trainer=CapturedTrainer
        worker=FrontierWorker(profile,output/'experiments')
        started=time.perf_counter()
        worker.start(config,FrontierConfig(recovery_interval=config.checkpoint_interval,profiling=profiling))
    try:
        while True:
            snapshot=worker.latest()
            if snapshot and snapshot.status=='Paused':break
            if snapshot and snapshot.status in ('Failed','Save failed'):raise RuntimeError(snapshot.message)
            time.sleep(.01)
        stopped=time.perf_counter()
        trainer=capture['trainer']
        if trainer.total_steps!=steps:raise RuntimeError('Benchmark did not stop at its exact step budget')
        updates=int(next(iter(trainer.optimizer.state.values()))['step'].item()) if trainer.optimizer.state else 0
        active=stopped-capture['first_step']
        usage=process_usage()
        metrics=trainer.metrics()
        metrics.pop('elapsed',None)
        result=dict(mode=mode,step_budget=steps,configuration=config.to_dict(),initial_weights=capture['initial_weights'],
            final_weights=weights(trainer.online),target_weights=weights(trainer.target),trajectory_sha256=trajectory.hexdigest(),
            optimizer_updates=updates,wall_seconds=stopped-started,execution_seconds=active,
            steps_per_second=steps/active,optimizer_updates_per_second=updates/active,
            peak_ram_bytes=usage['peak_ram_bytes'],metrics=metrics,
            profiling=snapshot.profiling if mode=='frontier' else {},
            publications=snapshot.publications if mode=='frontier' else None)
    finally:worker.shutdown()
    output.mkdir(parents=True,exist_ok=True)
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('mode','steps_per_second','optimizer_updates_per_second','wall_seconds','peak_ram_bytes')},indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=100000)
    parser.add_argument('--mode',choices=('both','legacy','frontier'),default='both')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--profiling',action='store_true')
    args=parser.parse_args()
    if args.steps<32 or args.steps%32:parser.error('Use a positive step budget divisible by 32 (default 100000)')
    output=args.output or Path('.artifacts')/('v151-benchmark-'+uuid4().hex[:8])
    if args.mode!='both':
        run(args.mode,args.steps,output,args.profiling)
        return
    if args.profiling:parser.error('Run profiling separately from the matched unprofiled comparison')
    output.mkdir(parents=True,exist_ok=False)
    results={}
    for mode in ('legacy','frontier'):
        subprocess.run([sys.executable,__file__,'--mode',mode,'--steps',str(args.steps),'--output',str(output/mode)],check=True)
        results[mode]=json.loads((output/mode/'result.json').read_text())
    a,b=results['legacy'],results['frontier']
    results['comparison']=dict(same_initial_weights=a['initial_weights']==b['initial_weights'],
        identical_transitions=a['trajectory_sha256']==b['trajectory_sha256'],
        identical_online_weights=a['final_weights']==b['final_weights'],
        identical_target_weights=a['target_weights']==b['target_weights'],
        same_optimizer_update_count=a['optimizer_updates']==b['optimizer_updates'],
        throughput_ratio=b['steps_per_second']/a['steps_per_second'])
    (output/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results['comparison'],indent=2),flush=True)
    print('Results:',output/'comparison.json',flush=True)


if __name__=='__main__':main()
