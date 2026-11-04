"""CPU Frontier worker with one shared learner and ordered environment collectors."""
from collections import deque
from dataclasses import dataclass,replace,asdict
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
from ai.model_registry import identity
from ai.parallel import ParallelTrainer,Completion
from services.training_time import CompletionRate
from ai.model_library import ModelLibrary,FAMILIES
from uuid import uuid4


class FrontierHistory(HistoryStore):
    def __init__(self,root):
        super().__init__(root)
        self.pending=[]
        self._writer().execute('''CREATE TABLE IF NOT EXISTS episode_diagnostics(
            experiment_id TEXT,episode INTEGER,optimizer_updates INTEGER,mean_loss REAL,last_loss REAL,
            gradient_norm REAL,PRIMARY KEY(experiment_id,episode))''')
        self._writer().execute('''CREATE TABLE IF NOT EXISTS episode_environments(
            run_id TEXT,episode INTEGER,environment_id INTEGER,local_sequence INTEGER,
            PRIMARY KEY(run_id,episode))''')
        self.environments=[]

    def record(self,record,diagnostics,environment=0,sequence=None):
        values=(record,diagnostics)
        association=(record.run_id,record.episode,environment,record.episode-1 if sequence is None else sequence)
        for existing,old_association in zip(self.pending,self.environments):
            if (existing[0].run_id,existing[0].episode)==(record.run_id,record.episode):
                if existing!=values or association!=old_association:raise ValueError('Conflicting committed completion')
                return
        self.pending.append((record,diagnostics))
        self.environments.append(association)
        if len(self.pending)>=25:self.flush()

    def flush(self):
        if self.pending:
            with self._writer() as db:
                for (record,diagnostics),association in zip(self.pending,self.environments):
                    values=tuple(getattr(record,k) for k in tuple(record.__dataclass_fields__)[:11])
                    old=db.execute('SELECT * FROM episodes WHERE run_id=? AND episode=?',values[:2]).fetchone()
                    if old is not None and old!=values:raise ValueError('History conflict for committed completion')
                    old_diagnostics=db.execute('SELECT * FROM episode_diagnostics WHERE experiment_id=? AND episode=?',values[:2]).fetchone()
                    old_association=db.execute('SELECT * FROM episode_environments WHERE run_id=? AND episode=?',values[:2]).fetchone()
                    if old_diagnostics is not None and old_diagnostics!=diagnostics:raise ValueError('Conflicting completion diagnostics')
                    if old_association is not None and old_association!=association:raise ValueError('Conflicting environment association')
                    db.execute('INSERT OR IGNORE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?)',values)
                    db.execute('INSERT OR IGNORE INTO episode_diagnostics VALUES (?,?,?,?,?,?)',diagnostics)
                db.executemany('INSERT OR IGNORE INTO episode_environments VALUES (?,?,?,?)',self.environments)
            self.pending.clear()
            self.environments.clear()
        super().flush()

    def emergency_save(self,root):
        if self.pending:
            atomic_json(Path(root)/'pending-completions.json',{'schema_version':1,
                'records':[asdict(record) for record,_ in self.pending],
                'diagnostics':[diagnostics for _,diagnostics in self.pending],'environments':self.environments})


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

    def start(self,config=TrainingConfig(),execution=FrontierConfig(),architecture=None,network=None,resume=None,name='',generation=None):
        if self.thread and self.thread.is_alive():raise ValueError('A Frontier session is already active')
        if not isinstance(config,TrainingConfig) or not isinstance(execution,FrontierConfig):raise ValueError('Invalid Frontier configuration')
        self.thread=threading.Thread(target=self._run,args=(config,execution,architecture,network,resume,name,generation),name='snake-frontier',daemon=True)
        self.thread.start()

    def command(self,command,value=None):self.commands.put((command,value))

    def latest(self):
        with self.lock:return self.snapshot

    def shutdown(self):
        if self.thread and self.thread.is_alive():
            self.command('shutdown')
            self.thread.join(timeout=30)
            if self.thread.is_alive():raise RuntimeError('Frontier is still reaching a safe checkpoint boundary; wait before closing')

    def _run(self,config,execution,architecture,network,resume,name,generation):
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
        completion_rate=CompletionRate()
        session_started=time.perf_counter()
        recorded=set()
        library=None
        lease=None;generation_id=None
        training_lease=True

        def drain(completions):
            for finished in completions:
                if finished.index in recorded:continue
                record=EpisodeRecord(store.identifier,finished.index,finished.score,finished.length,finished.reward,
                    finished.reason,finished.steps,finished.epsilon,trainer.elapsed,finished.score,finished.seed,
                    environment_id=finished.environment,local_sequence=finished.sequence)
                # Queue first; a flush failure retains the queue for idempotent retry.
                journal.record(record,(store.identifier,finished.index,finished.updates,
                    finished.mean_loss,finished.loss,finished.gradient_norm),finished.environment,finished.sequence)
                recorded.add(finished.index);records.append(record)
                if finished.loss is not None:losses.append(finished.loss)

        def flush():
            if journal:journal.flush()

        def save(kind='recovery'):
            nonlocal checkpoint,model
            flush()
            with profiler.measure('checkpoint_serialization'):
                folder,published=store.save(trainer,kind,execution.recovery_slots)
            library.index_checkpoint(folder,generation=generation_id)
            checkpoint=str(folder)
            if published:model=str(published)
            experiments.link(store.identifier,'checkpoint',folder)
            return folder

        def publish(include_charts=False):
            nonlocal publications,charts,last_cpu,last_sample,published_steps,lease,training_lease
            if generation_id and lease:
                if status=='Completed' and training_lease:
                    library.release(lease)
                    cp=library.find_checkpoint(checkpoint) if checkpoint else None
                    lease=library.acquire(generation_id,'Completed Frontier session',cp['id'] if cp else None)
                    training_lease=False
                library.status(generation_id,{'Running':'Training','Save failed':'Failed'}.get(status,status),lease)
                if trainer:library.progress(generation_id,trainer.completed,trainer.total_steps)
            with profiler.measure('metrics_aggregation'):
                metrics=trainer.metrics() if trainer else {}
                if trainer:
                    recent=list(trainer.scores)[-100:]
                    lengths=list(trainer.lengths)[-100:]
                    seconds=max(active,1e-9)
                    completed=trainer.completed-started_episodes
                    eps=completed/seconds
                    usage=process_usage()
                    if isinstance(trainer,ParallelTrainer) and trainer.pool.processes and status!='Failed':
                        children=trainer.pool.usage()
                        usage['cpu_seconds']+=sum(child['cpu_seconds'] for child in children)
                        for key in ('ram_bytes','peak_ram_bytes'):
                            usage[key]=usage[key]+sum(child[key] for child in children) if usage[key] is not None and all(child[key] is not None for child in children) else None
                    cpu=max(0.,100*(usage['cpu_seconds']-last_cpu['cpu_seconds'])/(active-last_sample)) if active>last_sample else None
                    metrics.update(target=trainer.config.episodes,progress=100*trainer.completed/trainer.config.episodes,
                        rolling_mean=fmean(recent) if recent else None,rolling_median=median(recent) if recent else None,
                        mean_length=fmean(lengths) if lengths else None,median_length=median(lengths) if lengths else None,
                        steps_per_second=(trainer.total_steps-started_steps)/seconds,episodes_per_second=eps,
                        optimizer_updates=trainer.optimizer_updates,optimizer_updates_per_second=(trainer.optimizer_updates-started_updates)/seconds,
                        session_wall_seconds=time.perf_counter()-session_started,
                        eta=completion_rate.remaining(trainer.config.episodes-trainer.completed),
                        environments=execution.environments,strategy=execution.resolved_strategy,
                        active_environments=trainer.active_environments if isinstance(trainer,ParallelTrainer) else int(not trainer.at_boundary),
                        update_policy=execution.update_policy,worker_health=('Failed' if status=='Failed' else 'Closed' if status in ('Stopped','Completed') else 'Healthy'),
                        partial_episode_recovery=isinstance(trainer,ParallelTrainer),
                        cpu_percent=cpu,ram_bytes=usage['ram_bytes'],peak_ram_bytes=usage['peak_ram_bytes'])
                    last_cpu,last_sample=usage,active
                    if include_charts:
                        if trainer.total_steps!=published_steps:
                            throughput.append((active,metrics['steps_per_second']))
                        # Refresh frequency is independent of record collection. Bound render/stat work.
                        dataset=Dataset(store.identifier,name,tuple(records),architecture=trainer.online.config.sizes,
                            model_identity=identity(trainer.online.config))
                        charts={mode:aggregate.prepare((dataset,),mode) for mode in ('distribution','mean_median')}
                        published_steps=trainer.total_steps
                    inference_seconds=profiler.seconds.get('neural_inference',0.)
                    metrics['inferences_per_second']=(profiler.calls.get('neural_inference',0)/inference_seconds
                        if profiler.enabled and inference_seconds else None)
            with profiler.measure('ui_synchronization'):
                publications+=1
                snapshot=FrontierSnapshot(status,message,metrics,charts,tuple(losses),tuple(throughput),checkpoint,model,
                    str(store.root) if store else None,profiler.report() if profiler.enabled else {},publications)
                with self.lock:self.snapshot=snapshot

        try:
            library=ModelLibrary(self.profile);library.scan()
            if resume:
                trainer,metadata=CheckpointStore(self.profile/'checkpoints').resume(resume)
                if config.episodes<trainer.completed:raise ValueError('Target must include completed checkpoint episodes')
                trainer.config=replace(trainer.config,episodes=config.episodes)
                trainer.environment.config=trainer.config
                config=trainer.config
                checkpoint=str(resume)
            elif execution.environments>1 or execution.resolved_strategy=='multiprocessing' or execution.update_policy=='vector':
                trainer=ParallelTrainer(config,execution,network,architecture)
            else:trainer=Trainer(config,network,architecture)
            if isinstance(trainer,ParallelTrainer):
                execution=replace(execution,environments=trainer.execution.environments,
                    strategy=trainer.execution.strategy,update_policy=trainer.execution.update_policy)
            elif resume:execution=replace(execution,environments=1,strategy='vectorized',update_policy='transition')
            self._trainer=trainer
            run=uuid4().hex
            model_info=identity(trainer.online.config)
            family=next(f for f,c in FAMILIES.items() if c==model_info['model_configuration_id'])
            generation_id=library.register_run(run,family,self.output/run,'Frontier',generation=generation,resume=resume,
                description='' if name=='Snake DQN Frontier' else name,data={'training':config.to_dict()})
            lease=library.acquire(generation_id,'Frontier Training',training=True)
            name=library.get(generation_id)['name'];trainer.generation_id=generation_id
            trainer.generation_number=library.get(generation_id)['number']
            trainer.training_interface='Frontier'
            library.allow_root(self.output)
            store=FrontierStore(self.output,self.profile,config,execution,name,identifier=run,resumed_from=resume)
            store.metadata.update(generation_id=generation_id,generation_number=library.get(generation_id)['number'],family=family)
            store.metadata['architecture']=trainer.online.config.to_dict()
            model_info=identity(trainer.online.config)
            store.metadata.update(model_info,input_count=trainer.online.config.input_size)
            import platform
            store.metadata.update(resolved_strategy=execution.resolved_strategy,seed_derivation='sha256:snake-frontier-seeds-v1/master/environment/local-sequence' if isinstance(trainer,ParallelTrainer) else 'legacy owned environment RNG',
                scheduling='synchronous, ascending environment ID; completions in transition order',
                target_sync_basis='global environment transitions',epsilon_basis='global environment transitions',
                python_version=platform.python_version(),torch_version=str(torch.__version__),
                resume_strategy='Exact quiescent vector boundary; partial episodes reconstructed from seed and action trace' if isinstance(trainer,ParallelTrainer) else store.metadata['resume_strategy'])
            atomic_json(store.root/'experiment.json',store.metadata)
            journal=FrontierHistory(store.root/'history')
            journal.register(store.identifier,name,trainer.online.config.sizes,
                dict(training=config.to_dict(),checkpoint_run_id=store.identifier,experiment_id=store.identifier,
                     observation_encoder=model_info['encoder_id'],model_identity=model_info,execution=execution.to_dict(),frontier=True,resumed_from=str(resume) if resume else None))
            experiments.save(Experiment(store.identifier,name,model_info['encoder_id'],trainer.online.config.input_size,trainer.online.config.sizes,
                config.to_dict(),{key:getattr(config,key) for key in ('food_reward','collision_reward','step_reward','stagnation_reward')},
                config.seed,tuple(range(25)),config.episodes*config.episode_cap,(store.identifier,),(),()))
            original_reset=trainer.environment.reset
            def reset(seed):
                nonlocal episode_seed
                episode_seed=seed
                return original_reset(seed)
            trainer.environment.reset=reset
            profiler.attach(trainer)
            updates=trainer.optimizer_updates
            started_steps,started_episodes,started_updates=trainer.total_steps,trainer.completed,updates
            completion_rate.update(trainer.completed,active)
            message=f'CPU / {execution.environments} environments / {execution.resolved_strategy} / updates per {execution.update_policy}'
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
                            if command=='stop' and isinstance(trainer,ParallelTrainer):trainer.close(measure=True)
                            flush();journal.status(store.identifier,status);publish(True)
                        elif command=='resume':
                            if trainer.completed>=trainer.config.episodes:
                                message='Episode target already completed; configure a new experiment to extend it'
                            else:
                                if isinstance(trainer,ParallelTrainer):trainer.restart_collectors()
                                status='Running';message=f'CPU / {execution.environments} environments / {execution.resolved_strategy}';journal.status(store.identifier,status)
                            publish()
                        elif command in ('save','save_stop'):
                            pending=True
                            if status=='Save failed':status='Running'
                            if command=='save_stop':finish_stop=True;status='Running'
                            message='Save at quiescent vector boundary (partial episodes retained)' if isinstance(trainer,ParallelTrainer) else 'Save pending at the next completed-episode boundary; resume if paused'
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
                        if finish_stop:
                            status='Stopped';finish_stop=False
                            if isinstance(trainer,ParallelTrainer):trainer.close(measure=True)
                        message='Resumable checkpoint saved: '+checkpoint
                    except Exception as error:
                        status='Paused';message='Checkpoint save failed; in-memory state retained: '+str(error);pending=False;finish_stop=False
                    publish(True)
                if trainer.completed>=trainer.config.episodes:
                    if status not in ('Completed','Save failed'):
                        try:
                            save('final')
                            status='Completed';message='Final model saved: '+model
                            if isinstance(trainer,ParallelTrainer):trainer.close(measure=True)
                            journal.status(store.identifier,status)
                        except Exception as error:status='Save failed';message='Final save failed; use Save Checkpoint to retry: '+str(error)
                        publish(True)
                if status!='Running':
                    time.sleep(.01)
                    continue
                if isinstance(trainer,ParallelTrainer):
                    milestone=min((n for n in execution.milestones if n>trainer.completed),default=config.episodes)
                    trainer.train_batch(completion_limit=milestone)
                    completions=trainer.completions
                else:
                    transition=trainer.train_step()
                    if len(trainer.replay)>=config.batch_size:
                        episode_updates+=1;episode_loss+=trainer.loss
                    completions=[]
                    if transition and transition.done:
                        env=trainer.environment
                        completions=[Completion(trainer.completed,0,trainer.completed-1,episode_seed,
                            env.game.score,env.steps,env.total_reward,transition.reason,trainer.total_steps,trainer.epsilon,
                            episode_updates,episode_loss/episode_updates if episode_updates else None,trainer.loss,trainer.gradient_norm)]
                        episode_updates=0;episode_loss=0.
                updates=trainer.optimizer_updates
                if completions:completion_rate.update(trainer.completed,active+time.perf_counter()-accounted)
                drain(completions)
                if completions:
                    previous=trainer.completed-len(completions)
                    if trainer.completed<config.episodes:
                        try:
                            if any(previous<episode<=trainer.completed for episode in execution.milestones):
                                flush()
                                with profiler.measure('checkpoint_serialization'):
                                    folder,published=store.snapshot(trainer)
                                checkpoint,model=str(folder),str(published)
                                library.index_checkpoint(folder,generation=generation_id)
                                experiments.link(store.identifier,'checkpoint',folder)
                            if trainer.completed//execution.recovery_interval>previous//execution.recovery_interval:save()
                        except Exception as error:
                            status='Paused';message='Automatic save failed; state retained: '+str(error);publish(True)
                    if trainer.completed//execution.refresh_episodes>previous//execution.refresh_episodes:
                        flush();publish(True)
                if not execution.headless:
                    # Optional paced headless mode, never a board/sound presentation.
                    time.sleep(1/9)
            journal.status(store.identifier,status if status=='Completed' else 'Stopped')
        except Exception as error:
            status='Failed';message=str(error)
            if trainer:updates=trainer.optimizer_updates
            if isinstance(trainer,ParallelTrainer) and journal:
                try:drain(trainer.completions);flush()
                except Exception as history_error:
                    message+=' / history persistence failed: '+str(history_error)
                    try:journal.emergency_save(store.root);message+=' / committed completions saved to emergency journal'
                    except Exception as emergency_error:message+=' / emergency journal unavailable: '+str(emergency_error)
            if isinstance(trainer,ParallelTrainer) and store:
                newly_saved=False
                if trainer.recovery_safe:
                    try:save();newly_saved=True;message+=' / new verified recovery saved: '+checkpoint
                    except Exception as recovery_error:message+=' / new recovery unavailable: '+str(recovery_error)
                else:message+=' / interrupted batch is unsafe; no rollback claimed'
                if not newly_saved:
                    if checkpoint:message+=' / previous verified recovery: '+checkpoint
                    else:message+=' / recovery unavailable: no verified checkpoint'
            if journal:
                try:flush();journal.status(store.identifier,status)
                except Exception:pass
            publish()
        finally:
            if isinstance(trainer,ParallelTrainer):trainer.close()
            if journal:
                try:journal.close()
                except Exception as error:
                    status='Failed';message+=' / history flush failed: '+str(error)
                    try:journal.emergency_save(store.root)
                    except Exception as emergency_error:message+=' / emergency journal unavailable: '+str(emergency_error)
                    publish()
            if store:
                try:
                    atomic_json(store.root/'session.json',dict(status=status,message=message,profiling=profiler.report(),
                        metrics=self.latest().metrics if self.latest() else {},optimizer_updates=updates,
                        wall_seconds=time.perf_counter()-session_started))
                finally:store.close()
            if library:library.release(lease)
