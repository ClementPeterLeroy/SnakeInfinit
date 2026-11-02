"""CPU / one environment. Existing Trainer owns every learning operation."""
from collections import deque
from dataclasses import dataclass,replace
from pathlib import Path
import queue
from statistics import fmean,median
import threading
import time
from ai.analysis_stats import Dataset,EpisodeRecord
from ai.analysis_model import AnalysisModel
from ai.checkpoints import CheckpointStore
from ai.dqn import Trainer
from ai.frontier_config import FrontierConfig
from ai.frontier_store import FrontierStore,atomic_json
from ai.experiments import Experiment,ExperimentStore,BASELINE_ENCODER
from ai.history import HistoryStore
from ai.profiling import Profiler,process_usage
from ai.training_config import TrainingConfig


class FrontierHistory(HistoryStore):
    def __init__(self,root):
        super().__init__(root)
        self.pending=[]
        self._writer().execute('''CREATE TABLE IF NOT EXISTS episode_diagnostics(
            experiment_id TEXT,episode INTEGER,optimizer_updates INTEGER,mean_loss REAL,last_loss REAL,
            gradient_norm REAL,PRIMARY KEY(experiment_id,episode))''')

    def record(self,record,diagnostics):
        self.pending.append((record,diagnostics))
        if len(self.pending)>=25:self.flush()

    def flush(self):
        if self.pending:
            with self._writer() as db:
                db.executemany('INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    [tuple(getattr(record,k) for k in record.__dataclass_fields__) for record,_ in self.pending])
                db.executemany('INSERT INTO episode_diagnostics VALUES (?,?,?,?,?,?)',[values for _,values in self.pending])
            self.pending.clear()
        super().flush()


@dataclass(frozen=True)
class FrontierSnapshot:
    status: str
    message: str
    metrics: dict
    charts: dict | None
    loss_trend: tuple
    throughput: tuple
    checkpoint: str | None
    model: str | None
    experiment: str | None
    profiling: dict
    publications: int


class FrontierWorker:
    def __init__(self,profile,output):
        self.profile,self.output=Path(profile),Path(output)
        self.commands=queue.Queue()
        self.lock=threading.Lock()
        self.snapshot=None
        self.thread=None
        self._trainer=None  # Worker-owned; never accessed by presentation code.

    def start(self,config=TrainingConfig(),execution=FrontierConfig(),architecture=None,network=None,resume=None,name='Snake DQN Frontier'):
        if self.thread and self.thread.is_alive():raise ValueError('A Frontier session is already active')
        if not isinstance(config,TrainingConfig) or not isinstance(execution,FrontierConfig):raise ValueError('Invalid Frontier configuration')
        self.thread=threading.Thread(target=self._run,args=(config,execution,architecture,network,resume,name),name='snake-frontier',daemon=True)
        self.thread.start()

    def command(self,command,value=None):self.commands.put((command,value))

    def latest(self):
        with self.lock:return self.snapshot

    def shutdown(self):
        if self.thread and self.thread.is_alive():
            self.command('shutdown')
            self.thread.join(timeout=30)
            if self.thread.is_alive():raise RuntimeError('Frontier is still reaching a safe checkpoint boundary; wait before closing')

    def _run(self,config,execution,architecture,network,resume,name):
        import torch
        torch.set_num_threads(1)
        store=journal=trainer=None
        experiments=ExperimentStore(self.profile/'experiments')
        profiler=Profiler(execution.profiling)
        status,message='Running','Initializing Frontier / CPU / 1 environment'
        checkpoint=model=None
        pending=finish_stop=False
        records=deque(maxlen=20000)
        losses=deque(maxlen=2000)
        throughput=deque(maxlen=1000)
        publications=0
        active=0.
        episode_seed=None
        episode_updates=0
        episode_loss=0.
        started_steps=started_episodes=started_updates=0
        published_steps=0
        updates=0
        last_cpu=process_usage()
        last_sample=0.
        charts=None
        aggregate=AnalysisModel()

        def flush():
            if journal:journal.flush()

        def save(kind='recovery'):
            nonlocal checkpoint,model
            flush()
            with profiler.measure('checkpoint_serialization'):
                folder,published=store.save(trainer,kind,execution.recovery_slots)
            checkpoint=str(folder)
            if published:model=str(published)
            experiments.link(store.identifier,'checkpoint',folder)
            return folder

        def publish(include_charts=False):
            nonlocal publications,charts,last_cpu,last_sample,published_steps
            with profiler.measure('metrics_aggregation'):
                metrics=trainer.metrics() if trainer else {}
                if trainer:
                    recent=list(trainer.scores)[-100:]
                    lengths=list(trainer.lengths)[-100:]
                    seconds=max(active,1e-9)
                    completed=trainer.completed-started_episodes
                    eps=completed/seconds
                    usage=process_usage()
                    cpu=100*(usage['cpu_seconds']-last_cpu['cpu_seconds'])/(active-last_sample) if active>last_sample else None
                    metrics.update(target=trainer.config.episodes,progress=100*trainer.completed/trainer.config.episodes,
                        rolling_mean=fmean(recent) if recent else None,rolling_median=median(recent) if recent else None,
                        mean_length=fmean(lengths) if lengths else None,median_length=median(lengths) if lengths else None,
                        steps_per_second=(trainer.total_steps-started_steps)/seconds,episodes_per_second=eps,
                        optimizer_updates=updates,optimizer_updates_per_second=(updates-started_updates)/seconds,
                        eta=(trainer.config.episodes-trainer.completed)/eps if eps else None,
                        cpu_percent=cpu,ram_bytes=usage['ram_bytes'],peak_ram_bytes=usage['peak_ram_bytes'])
                    last_cpu,last_sample=usage,active
                    if include_charts:
                        if trainer.total_steps!=published_steps:
                            throughput.append((active,metrics['steps_per_second']))
                        # Refresh frequency is independent of record collection. Bound render/stat work.
                        dataset=Dataset(store.identifier,name,tuple(records),architecture=trainer.online.config.sizes)
                        charts={mode:aggregate.prepare((dataset,),mode) for mode in ('distribution','mean_median')}
                        published_steps=trainer.total_steps
            with profiler.measure('ui_synchronization'):
                publications+=1
                snapshot=FrontierSnapshot(status,message,metrics,charts,tuple(losses),tuple(throughput),checkpoint,model,
                    str(store.root) if store else None,profiler.report() if profiler.enabled else {},publications)
                with self.lock:self.snapshot=snapshot

        try:
            if resume:
                trainer,metadata=CheckpointStore(self.profile/'checkpoints').resume(resume)
                if config.episodes<trainer.completed:raise ValueError('Target must include completed checkpoint episodes')
                trainer.config=replace(trainer.config,episodes=config.episodes)
                trainer.environment.config=trainer.config
                config=trainer.config
            else:trainer=Trainer(config,network,architecture)
            self._trainer=trainer
            store=FrontierStore(self.output,self.profile,config,execution,name,resumed_from=resume)
            store.metadata['architecture']=trainer.online.config.to_dict()
            atomic_json(store.root/'experiment.json',store.metadata)
            journal=FrontierHistory(store.root/'history')
            journal.register(store.identifier,name,trainer.online.config.sizes,
                dict(training=config.to_dict(),checkpoint_run_id=store.identifier,experiment_id=store.identifier,
                     observation_encoder='snake-classic-binary11-v1',frontier=True,resumed_from=str(resume) if resume else None))
            experiments.save(Experiment(store.identifier,name,BASELINE_ENCODER,11,trainer.online.config.sizes,
                config.to_dict(),{key:getattr(config,key) for key in ('food_reward','collision_reward','step_reward','stagnation_reward')},
                config.seed,tuple(range(25)),config.episodes*config.episode_cap,(store.identifier,),(),()))
            original_reset=trainer.environment.reset
            def reset(seed):
                nonlocal episode_seed
                episode_seed=seed
                return original_reset(seed)
            trainer.environment.reset=reset
            profiler.attach(trainer)
            updates=max(0,trainer.total_steps-trainer.config.batch_size+1)
            started_steps,started_episodes,started_updates=trainer.total_steps,trainer.completed,updates
            publish(True)
            accounted=time.perf_counter()
            while True:
                now=time.perf_counter()
                if status=='Running':
                    elapsed=now-accounted
                    active+=elapsed;trainer.elapsed+=elapsed
                accounted=now
                shutdown=False
                # Commands remain responsive, without per-step model copies or locks.
                if trainer.total_steps%32==0 or trainer.at_boundary or status!='Running' or not execution.headless:
                    while True:
                        try:command,value=self.commands.get_nowait()
                        except queue.Empty:break
                        if command=='shutdown':shutdown=True
                        elif command in ('pause','stop'):
                            status='Paused' if command=='pause' else 'Stopped'
                            message='In-memory state retained, including the current episode'
                            flush();journal.status(store.identifier,status);publish(True)
                        elif command=='resume':
                            if trainer.completed>=trainer.config.episodes:
                                message='Episode target already completed; configure a new experiment to extend it'
                            else:
                                status='Running';message='CPU / 1 environment / unchanged DQN';journal.status(store.identifier,status)
                            publish()
                        elif command in ('save','save_stop'):
                            pending=True
                            if status=='Save failed':status='Running'
                            if command=='save_stop':finish_stop=True;status='Running'
                            message='Save pending at the next completed-episode boundary; resume if paused'
                            publish()
                if shutdown:
                    flush()
                    if status not in ('Completed','Failed','Save failed'):
                        status='Stopped'
                        message='Worker closed at a safe step; last message: '+message
                        publish(True)
                    break
                if pending and trainer.at_boundary:
                    try:
                        save();pending=False
                        if finish_stop:status='Stopped';finish_stop=False
                        message='Resumable checkpoint saved: '+checkpoint
                    except Exception as error:
                        status='Paused';message='Checkpoint save failed; in-memory state retained: '+str(error);pending=False;finish_stop=False
                    publish(True)
                if trainer.completed>=trainer.config.episodes:
                    if status not in ('Completed','Save failed'):
                        try:
                            save('final')
                            status='Completed';message='Final model saved: '+model
                            journal.status(store.identifier,status)
                        except Exception as error:status='Save failed';message='Final save failed; use Save Checkpoint to retry: '+str(error)
                        publish(True)
                if status!='Running':
                    time.sleep(.01)
                    continue
                transition=trainer.train_step()
                if len(trainer.replay)>=config.batch_size:
                    updates+=1;episode_updates+=1;episode_loss+=trainer.loss
                if transition and transition.done:
                    env=trainer.environment
                    record=EpisodeRecord(store.identifier,trainer.completed,env.game.score,env.steps,env.total_reward,
                        transition.reason,trainer.total_steps,trainer.epsilon,trainer.elapsed+(time.perf_counter()-accounted),env.game.apples_eaten,episode_seed)
                    records.append(record)
                    journal.record(record,(store.identifier,trainer.completed,episode_updates,
                        episode_loss/episode_updates if episode_updates else None,trainer.loss,trainer.gradient_norm))
                    if trainer.loss is not None:losses.append(trainer.loss)
                    episode_updates=0;episode_loss=0.
                    if trainer.completed<config.episodes:
                        try:
                            if trainer.completed in execution.milestones:
                                flush()
                                with profiler.measure('checkpoint_serialization'):
                                    folder,published=store.snapshot(trainer)
                                checkpoint,model=str(folder),str(published)
                                experiments.link(store.identifier,'checkpoint',folder)
                            if trainer.completed%execution.recovery_interval==0:save()
                        except Exception as error:
                            status='Paused';message='Automatic save failed; state retained: '+str(error);publish(True)
                    if trainer.completed%execution.refresh_episodes==0:
                        flush();publish(True)
                if not execution.headless:
                    # Optional paced headless mode, never a board/sound presentation.
                    time.sleep(1/9)
            journal.status(store.identifier,status if status=='Completed' else 'Stopped')
        except Exception as error:
            status='Failed';message=str(error)
            if journal:
                try:flush();journal.status(store.identifier,status)
                except Exception:pass
            publish()
        finally:
            if journal:
                try:journal.close()
                except Exception as error:
                    status='Failed';message='History flush failed: '+str(error);publish()
            if store:
                try:
                    atomic_json(store.root/'session.json',dict(status=status,message=message,profiling=profiler.report(),
                        metrics=trainer.metrics() if trainer else {},optimizer_updates=updates))
                finally:store.close()
