"""Audit-only CPU matrix with warmup, repetitions and exclusive stage timings.

Uses existing collectors and learners. All instrumentation lives in this process
or explicitly owned diagnostic children; no production modules/files are edited.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class Timers:
    def __init__(self):self.seconds={};self.calls={};self.stack=[]

    @contextmanager
    def measure(self,name):
        frame=[time.perf_counter(),0.];self.stack.append(frame)
        try:yield
        finally:
            elapsed=time.perf_counter()-frame[0];self.stack.pop()
            if self.stack:self.stack[-1][1]+=elapsed
            self.seconds[name]=self.seconds.get(name,0.)+max(0.,elapsed-frame[1])
            self.calls[name]=self.calls.get(name,0)+1

    def wrap(self,obj,method,name):
        original=getattr(obj,method)
        def measured(*args,**kwargs):
            with self.measure(name):return original(*args,**kwargs)
        setattr(obj,method,measured)
        return original

    def report(self):return {name:dict(seconds=seconds,calls=self.calls[name]) for name,seconds in self.seconds.items()}


def instrument_episode(episode,timers):
    timers.wrap(episode,'observation','observation_encoding')
    timers.wrap(episode,'step','environment_dynamics')
    timers.wrap(episode,'reset','environment_reset')
    return episode


def audit_worker_entry(connection,config,encoder):
    """Preserve the production protocol; augment usage replies with child timers."""
    from ai import environment_pool,profiling
    timer=Timers();make=environment_pool.make_episode;usage=profiling.process_usage
    environment_pool.make_episode=lambda config,encoder:instrument_episode(make(config,encoder),timer)
    profiling.process_usage=lambda:{**usage(),'audit_timers':timer.report()}
    environment_pool.worker_entry(connection,config,encoder)


def difference(after,before):
    return {name:dict(seconds=value['seconds']-before.get(name,{}).get('seconds',0.),
        calls=value['calls']-before.get(name,{}).get('calls',0)) for name,value in after.items()}


def trial(architecture,count,strategy,mode,steps,warmup,seed):
    import torch
    from ai import environment_pool
    from ai.dqn import Trainer
    from ai.parallel import ParallelTrainer
    from ai.model_registry import model_spec
    from ai.frontier_config import FrontierConfig
    from ai.training_config import TrainingConfig
    from ai.profiling import process_usage
    # All cells use the same settings, including a documented post-decay epsilon.
    config=TrainingConfig(episodes=1000000,seed=seed,epsilon_start=.05,epsilon_min=.05)
    spec=model_spec(architecture,seed)
    execution=FrontierConfig(environments=count,strategy=strategy,update_policy='transition')
    timer=Timers();collector=trainer=None
    old_entry=environment_pool.worker_entry
    environment_pool.worker_entry=audit_worker_entry
    restore=[]
    try:
        started=time.perf_counter()
        if mode=='environment':collector=environment_pool.EnvironmentPool(config,spec.encoder_id,count,strategy)
        else:
            trainer=Trainer(config,architecture=spec) if count==1 and strategy=='vectorized' else ParallelTrainer(config,execution,architecture=spec)
            collector=getattr(trainer,'pool',None)
        startup=time.perf_counter()-started
        if collector:
            for episode in collector.episodes:instrument_episode(episode,timer)
            timer.wrap(collector,'step','collection_dispatch')
            timer.wrap(collector,'reset','reset_dispatch')
            if collector.processes:timer.wrap(collector,'_receive','receive_wait_and_deserialize')
        if trainer:
            if collector is None:instrument_episode(trainer.environment,timer)
            choosing=[False]
            choose_method='choose_actions' if hasattr(trainer,'choose_actions') else 'choose_action'
            choose=getattr(trainer,choose_method)
            def choose_timed(*args,**kwargs):
                choosing[0]=True
                try:
                    with timer.measure('policy_selection'):return choose(*args,**kwargs)
                finally:choosing[0]=False
            setattr(trainer,choose_method,choose_timed)
            original_forward=trainer.online.forward
            def forward(*args,**kwargs):
                with timer.measure('policy_forward' if choosing[0] else 'training_online_forward'):
                    return original_forward(*args,**kwargs)
            trainer.online.forward=forward
            timer.wrap(trainer.target,'forward','target_forward')
            timer.wrap(trainer.target,'load_state_dict','target_synchronization')
            timer.wrap(trainer.replay,'sample','replay_sampling')
            timer.wrap(trainer.optimizer,'step','optimizer_step')
            if trainer.codec:timer.wrap(trainer.codec,'stack','observation_batching')
            for obj,method,name in ((torch,'tensor','tensor_construction'),(torch.autograd,'backward','backpropagation'),
                (torch.nn.utils,'clip_grad_norm_','gradient_clipping')):
                restore.append((obj,method,timer.wrap(obj,method,name)))
        rngs=[random.Random(environment_pool.episode_seed(seed,i,2**32)) for i in range(count)]
        sequences=[0]*count;lengths=[0]*count;completed_lengths=[]
        per_environment=[0]*count;active_min=count;active_max=0;batch_sizes=[];completed=0
        if mode=='environment':
            for i in range(count):collector.reset(i,environment_pool.episode_seed(seed,i,0))
        def advance(budget,record):
            nonlocal completed,active_min,active_max
            collected=0
            while collected<budget:
                width=min(count,budget-collected)
                if mode=='environment':
                    actions=[(i,rngs[i].randrange(3)) for i in range(width)]
                    results=collector.step(actions)
                    for (i,_),(transition,score,apples) in zip(actions,results):
                        lengths[i]+=1
                        if record:per_environment[i]+=1
                        if transition.done:
                            completed+=1;completed_lengths.append(lengths[i]);lengths[i]=0;sequences[i]+=1
                            collector.reset(i,environment_pool.episode_seed(seed,i,sequences[i]))
                    size=width
                elif hasattr(trainer,'train_batch'):
                    batch=trainer.train_batch(max_transitions=width);size=len(batch)
                    if record:
                        for i in range(size):per_environment[i]+=1
                else:
                    trainer.train_step();size=1
                    if record:per_environment[0]+=1
                collected+=size
                if record:
                    batch_sizes.append(size);active_min=min(active_min,size);active_max=max(active_max,size)
            return collected
        advance(warmup,False)
        parent_before=process_usage();children_before=collector.usage() if collector else []
        times_before=timer.report();updates_before=trainer.optimizer_updates if trainer else 0
        episodes_before=trainer.completed if trainer else completed
        timer_started=time.perf_counter();actual=advance(steps,True);elapsed=time.perf_counter()-timer_started
        times_after=timer.report()
        after=process_usage();children_after=collector.usage() if collector else []
        updates=(trainer.optimizer_updates-updates_before) if trainer else 0
        episodes=(trainer.completed if trainer else completed)-episodes_before
        parent_times=difference(times_after,times_before)
        child_times={}
        for before,after_child in zip(children_before,children_after):
            for name,value in difference(after_child['audit_timers'],before['audit_timers']).items():
                stage=child_times.setdefault(name,dict(seconds=0.,calls=0))
                stage['seconds']+=value['seconds'];stage['calls']+=value['calls']
        cpu=after['cpu_seconds']-parent_before['cpu_seconds']+sum(c['cpu_seconds'] for c in children_after)-sum(c['cpu_seconds'] for c in children_before)
        return dict(architecture=architecture,environments=count,strategy=strategy,mode=mode,seed=seed,warmup_steps=warmup,
            requested_steps=steps,actual_steps=actual,seconds=elapsed,steps_per_second=actual/elapsed,
            per_environment_steps=per_environment,per_environment_steps_per_second=[n/elapsed for n in per_environment],
            active_batch_min=active_min,active_batch_max=active_max,mean_batch_size=sum(batch_sizes)/len(batch_sizes),
            optimizer_updates=updates,updates_per_second=updates/elapsed,updates_per_transition=updates/actual,
            completed_episodes=episodes,episodes_per_second=episodes/elapsed,startup_seconds=startup,
            parent_cpu_and_children_percent=100*cpu/elapsed,parent_ram_bytes=after['ram_bytes'],parent_lifetime_peak_ram_bytes=after['peak_ram_bytes'],
            child_peak_ram_sum_bytes=sum(c['peak_ram_bytes'] or 0 for c in children_after),parent_timers=parent_times,child_service_timer_sums=child_times,
            profiling='Exclusive parent timings; child service wall-time sums are separate and may overlap parent receive waits',
            torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),failures=0)
    finally:
        for obj,method,original in reversed(restore):setattr(obj,method,original)
        if trainer and hasattr(trainer,'close'):trainer.close()
        elif collector:collector.close()
        environment_pool.worker_entry=old_entry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=256)
    parser.add_argument('--environment-steps',type=int,default=4096)
    parser.add_argument('--warmup',type=int,default=128)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--architectures',default='ABC')
    parser.add_argument('--strategies',default='vectorized,multiprocessing')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.steps<8 or args.environment_steps<8 or args.steps%8 or args.environment_steps%8 or args.warmup<64 or args.warmup%8 or args.repeats<1:parser.error('Use positive multiples of 8 and warmup >=64')
    if args.output.exists():parser.error('Choose an unused output path')
    import platform,torch
    torch.set_num_threads(1)
    report=dict(python=platform.python_version(),torch=str(torch.__version__),platform=platform.platform(),
        mode='audit instrumentation, no production changes',config=dict(learning_rate=.001,gamma=.99,batch_size=64,replay_capacity=100000,
        epsilon_start=.05,epsilon_min=.05,target_sync=1000,update_policy='transition',warmup=args.warmup,repetitions=args.repeats),
        caveats=['Timing instrumentation overhead included','No long-training replay occupancy or learned-policy quality claim','Child service times overlap parent receive waits'],trials=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for repeat in range(args.repeats):
        architectures=args.architectures if repeat%2==0 else args.architectures[::-1]
        for architecture in architectures:
            counts=(1,2,4,8) if repeat%2==0 else (8,4,2,1)
            for count in counts:
                for strategy in args.strategies.split(','):
                    for mode in ('environment','training'):
                        result=trial(architecture,count,strategy,mode,args.environment_steps if mode=='environment' else args.steps,args.warmup,7+repeat)
                        result['repetition']=repeat+1;report['trials'].append(result)
                        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
                        print(f"{repeat+1} {architecture} {count} {strategy} {mode}: {result['steps_per_second']:.1f} steps/s, {result['optimizer_updates']} updates",flush=True)


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
