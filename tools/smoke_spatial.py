"""Bounded, isolated real-Frontier training pilots; no production models or Git."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def run(configuration,steps,seconds,output):
    import torch
    torch.set_num_threads(1)
    from ai.dqn import Trainer
    from ai.frontier import FrontierWorker
    from ai.frontier_config import FrontierConfig
    from ai.model_registry import model_spec,identity
    from ai.training_config import TrainingConfig
    from ai.checkpoints import CheckpointStore
    from ai.persistence import ModelStore
    from ai.analysis_repository import AnalysisRepository
    from ai.analysis_export import export_named_csv
    from ai.analysis_stats import summary
    from ai.evaluation import evaluate_checkpoints
    from ai.profiling import process_usage
    import ai.frontier as module
    output=Path(output).resolve()
    capture={}
    target=max(32,steps//32*32)
    class Captured(Trainer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            digest=hashlib.sha256()
            for parameter in self.online.parameters():digest.update(bytes(parameter.detach().contiguous().view(torch.uint8).flatten().tolist()))
            capture['initial_weights']=digest.hexdigest()
        def train_step(self):
            if 'first_step' not in capture:capture['first_step']=time.perf_counter()
            transition=super().train_step()
            if self.total_steps%2048==0:print(configuration+f': {self.total_steps:,} steps',flush=True)
            if not capture.get('paused') and self.total_steps%32==0 and (self.total_steps>=target or time.perf_counter()-capture['first_step']>=seconds):
                capture['paused']=True;worker.command('pause')
            return transition
    module.Trainer=Captured
    profile=output/'profile'
    worker=FrontierWorker(profile,output/'experiments')
    config=TrainingConfig(episodes=1000000,seed=7)
    spec=model_spec(configuration,7)
    start=time.perf_counter()
    worker.start(config,FrontierConfig(profiling=True),architecture=spec,name='Smoke '+configuration)
    def wait(statuses,timeout):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            snapshot=worker.latest()
            if snapshot and snapshot.status in statuses:return snapshot
            if snapshot and snapshot.status in ('Failed','Save failed'):raise RuntimeError(snapshot.message)
            time.sleep(.01)
        raise RuntimeError('Smoke worker timed out')
    try:
        paused=wait(('Paused',),seconds+60)
        trained=time.perf_counter()
        duration=trained-capture['first_step']
        actual=paused.metrics['steps']
        worker.command('save_stop')
        saved=wait(('Stopped',),60)
        checkpoint=Path(saved.checkpoint)
        store=CheckpointStore(profile/'checkpoints')
        restored,metadata=store.resume(checkpoint)
        restored.train_step()
        if restored.loss is None or not torch.isfinite(torch.tensor(restored.loss)):raise RuntimeError('No finite optimizer loss after resume')
        frozen,_=store.load_model(checkpoint)
        portable=ModelStore(output/'portable')
        portable.save('model_'+configuration,frozen,reinforcement_learning=True)
        portable.load('model_'+configuration)
        repository=AnalysisRepository(profile)
        data=repository.load(next(c for c in repository.choices() if c.source==checkpoint))
        csv=export_named_csv(output/'exports',(data,))
        evaluation=evaluate_checkpoints([checkpoint],(0,1),profile/'evaluations')
        result=dict(configuration=configuration,identity=identity(spec),initial_weights=capture['initial_weights'],
            requested_steps=steps,actual_steps_at_pause=actual,actual_steps_after_boundary_save=saved.metrics['steps'],
            time_limit_seconds=seconds,execution_seconds=duration,total_validation_wall_seconds=time.perf_counter()-start,
            steps_per_second=actual/duration,optimizer_updates=paused.metrics['optimizer_updates'],
            optimizer_updates_per_second=paused.metrics['optimizer_updates']/duration,loss=paused.metrics['loss'],
            metrics=paused.metrics,summary=summary(data.records),profiling=paused.profiling,
            checkpoint=str(checkpoint),portable_model=str(portable.path('model_'+configuration)),csv=str(csv),
            evaluation=str(evaluation),resume_optimization_verified=True,peak_ram_bytes=process_usage()['peak_ram_bytes'])
        output.mkdir(parents=True,exist_ok=True)
        (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({key:result[key] for key in ('configuration','actual_steps_at_pause','execution_seconds','steps_per_second','optimizer_updates','loss')},indent=2),flush=True)
        return result
    finally:worker.shutdown()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',choices=('A','B','C','D','all'),default='all')
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--seconds',type=float,default=120.)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.steps<32 or not 0<args.seconds<=600:parser.error('Choose at least 32 steps and 0..600 seconds')
    output=args.output or Path('.artifacts')/('v152-smoke-'+uuid4().hex[:8])
    if args.configuration!='all':run(args.configuration,args.steps,args.seconds,output);return
    output.mkdir(parents=True,exist_ok=False)
    results={}
    for configuration in 'BCD':
        subprocess.run([sys.executable,__file__,'--configuration',configuration,'--steps',str(args.steps),'--seconds',str(args.seconds),'--output',str(output/configuration)],check=True)
        results[configuration]=json.loads((output/configuration/'result.json').read_text())
    if results['C']['initial_weights']!=results['D']['initial_weights']:raise RuntimeError('CNN initial weights do not match')
    (output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('Pilot results:',output/'results.json',flush=True)


if __name__=='__main__':main()
