"""Fixed-budget CPU collector/training benchmarks; writes only the requested report."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ai.frontier_config import FrontierConfig
from ai.environment_pool import EnvironmentPool,episode_seed
from ai.model_registry import model_spec
from ai.training_config import TrainingConfig
from ai.profiling import process_usage


def trial(architecture,count,strategy,category,policy,steps,updates):
    config=TrainingConfig(episodes=1000000,seed=7)
    spec=model_spec(architecture,7)
    execution=FrontierConfig(environments=count,strategy=strategy,update_policy=policy)
    start=time.perf_counter();collector=trainer=None
    sequences=[0]*count;lengths=[];episode_lengths=[0]*count
    try:
        if category=='environment':
            collector=EnvironmentPool(config,spec.encoder_id,count,strategy)
            for index in range(count):collector.reset(index,episode_seed(7,index,0))
            rngs=[random.Random(episode_seed(7,i,2**32)) for i in range(count)]
        else:
            from ai.parallel import ParallelTrainer
            from ai.dqn import Trainer
            trainer=(Trainer(config,architecture=spec) if count==1 and strategy=='vectorized' and policy=='transition'
                else ParallelTrainer(config,execution,architecture=spec))
        startup=time.perf_counter()-start
        pool=collector if collector else getattr(trainer,'pool',None)
        children_before=pool.usage() if pool else []
        before=process_usage();peak=before['ram_bytes'] or 0
        started=time.perf_counter();collected=0
        while True:
            if category=='environment':
                actions=[(i,rngs[i].randrange(3)) for i in range(min(count,steps-collected))]
                for (index,_),(transition,score,apples) in zip(actions,collector.step(actions)):
                    collected+=1;episode_lengths[index]+=1
                    if transition.done:
                        lengths.append(episode_lengths[index]);episode_lengths[index]=0;sequences[index]+=1
                        collector.reset(index,episode_seed(7,index,sequences[index]))
            elif hasattr(trainer,'train_batch'):
                limit=steps-trainer.total_steps if category=='equal_steps' else None
                if category=='equal_updates' and policy=='transition':
                    warmup=max(0,config.batch_size-1-trainer.total_steps)
                    limit=warmup+updates-trainer.optimizer_updates
                trainer.train_batch(max_transitions=limit)
                collected=trainer.total_steps
            else:trainer.train_step();collected=trainer.total_steps
            if collected%64==0:
                usage=process_usage();peak=max(peak,usage['ram_bytes'] or 0)
            if category=='equal_updates':
                if trainer.optimizer_updates>=updates:break
            elif collected>=steps:break
        elapsed=time.perf_counter()-started
        usage=process_usage();peak=max(peak,usage['ram_bytes'] or 0)
        children_after=pool.usage() if pool else []
        child_cpu=sum(c['cpu_seconds'] for c in children_after)-sum(c['cpu_seconds'] for c in children_before)
        if trainer:lengths=list(trainer.lengths)
        return dict(architecture=architecture,environments=count,strategy=strategy,category=category,update_policy=policy,
            startup_seconds=startup,environment_steps=collected,seconds=elapsed,steps_per_second=collected/elapsed,
            optimizer_updates=trainer.optimizer_updates if trainer else 0,
            optimizer_updates_per_second=trainer.optimizer_updates/elapsed if trainer else 0,
            completed_episodes=trainer.completed if trainer else len(lengths),
            average_episode_length=sum(lengths)/len(lengths) if lengths else None,
            cpu_percent=100*(usage['cpu_seconds']-before['cpu_seconds']+child_cpu)/elapsed,
            child_peak_ram_bytes=sum(c['peak_ram_bytes'] or 0 for c in children_after),
            sampled_peak_parent_ram_bytes=peak,process_lifetime_peak_ram_bytes=usage['peak_ram_bytes'],worker_failures=0)
    finally:
        if collector:collector.close()
        if trainer and hasattr(trainer,'close'):trainer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=2048)
    parser.add_argument('--updates',type=int,default=256)
    parser.add_argument('--simulation-steps',type=int,default=4096)
    parser.add_argument('--architectures',default='ABCD')
    parser.add_argument('--counts',default='1,2,4,8')
    parser.add_argument('--strategies',default='vectorized,multiprocessing')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.steps<64 or args.updates<1 or args.simulation_steps<8:parser.error('Use steps >=64, updates >=1 and simulation steps >=8')
    counts=tuple(int(n) for n in args.counts.split(','));strategies=args.strategies.split(',')
    for count in counts:
        for strategy in strategies:FrontierConfig(environments=count,strategy=strategy)
    if any(a not in 'ABCD' for a in args.architectures):parser.error('Architectures must be A/B/C/D')
    if args.output.exists():parser.error('Output already exists; choose a new report name')
    import platform,torch
    torch.set_num_threads(1)
    report=dict(python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),
        seed=7,torch_threads=1,steps_budget=args.steps,updates_budget=args.updates,simulation_steps=args.simulation_steps,
        notes='Synchronous collection; CPU includes children; parent and child memory reported separately. Startup excluded, shutdown outside timing. No chart/file I/O in timed training. Short timings are noisy.',trials=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for architecture in args.architectures:
        for count in counts:
            for strategy in strategies:
                for category in ('environment','equal_steps','equal_updates'):
                    policies=('transition',) if category=='environment' else ('transition','vector')
                    for policy in policies:
                        try:result=trial(architecture,count,strategy,category,policy,args.simulation_steps if category=='environment' else args.steps,args.updates)
                        except Exception as error:
                            result=dict(architecture=architecture,environments=count,strategy=strategy,category=category,update_policy=policy,worker_failures=1,error=str(error))
                        report['trials'].append(result)
                        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
                        print(f"{architecture} {count} {strategy} {category} {policy}: "+(f"{result['steps_per_second']:.1f} steps/s, {result['optimizer_updates']} updates" if 'steps_per_second' in result else result['error']),flush=True)
    if any(r['worker_failures'] for r in report['trials']):raise SystemExit(1)


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
