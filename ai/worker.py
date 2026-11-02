"""Single owner of all training tensors; UI receives only detached immutable data."""
from dataclasses import dataclass
import queue
import threading
import time
from uuid import uuid4
import torch
from ai.training_config import TrainingConfig
from ai.checkpoints import CheckpointStore
from ai.dqn import Trainer
from ai.history import HistoryStore
from ai.analysis_stats import EpisodeRecord
from ai.experiments import ExperimentStore


@dataclass(frozen=True)
class TrainingSnapshot:
    status: str
    message: str
    metrics: tuple
    histories: tuple
    state: object
    mode: str
    multiplier: int
    steps_per_second: float
    episodes_per_second: float
    checkpoint: str | None
    pending_save: bool

    @property
    def values(self):
        return dict(self.metrics)


class TrainingWorker:
    def __init__(self, store=None):
        self.store = store or CheckpointStore()
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.snapshot = None
        self.thread = None

    def start(self, config=None, architecture=None, network=None, resume=None, name='Snake DQN', episode_target=None):
        if self.thread and self.thread.is_alive():
            raise ValueError("Close the current worker before starting another run")
        self.thread = threading.Thread(target=self._run,args=(config,architecture,network,resume,name,episode_target),
                                       name='snake-dqn',daemon=True)
        self.thread.start()

    def command(self, command, value=None):
        self.commands.put((command,value))

    def latest(self):
        with self.lock:
            return self.snapshot

    def shutdown(self):
        if self.thread and self.thread.is_alive():
            self.command('shutdown')
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError("Training worker did not reach a safe shutdown boundary")

    def _run(self,config,architecture,network,resume,name,episode_target):
        trainer = None
        status, message, mode, multiplier = 'Running','Initializing CPU DQN','realtime',1
        pending, checkpoint, shutdown = False,None,False
        active_time = 0.
        start_steps = start_episodes = 0
        last_publish, deadline = 0.,time.monotonic()
        run = uuid4().hex
        journal=HistoryStore(self.store.root.parent/'history')
        experiments=ExperimentStore(self.store.root.parent/'experiments')
        history_warning=None
        history_failed=False
        experiment_id=None
        history_run=None
        def history_call(operation,*args):
            nonlocal history_warning,history_failed
            is_journal=getattr(operation,'__self__',None) is journal
            if is_journal and history_failed and operation.__name__!='close':
                return None
            try:
                return operation(*args)
            except Exception as error:
                history_warning=str(error)
                if is_journal:history_failed=True
                return None
        def publish():
            nonlocal last_publish
            metrics = trainer.metrics() if trainer else {}
            seconds = max(active_time,1e-9)
            notice=message+(' | History warning: '+history_warning if history_warning else '')
            snap = TrainingSnapshot(status,notice,tuple(metrics.items()),
                tuple((key,tuple(getattr(trainer,key))) for key in ('scores','losses','epsilons')) if trainer else (),
                trainer.environment.game.get_state() if trainer else None,mode,multiplier,
                (trainer.total_steps-start_steps)/seconds if trainer else 0.,
                (trainer.completed-start_episodes)/seconds if trainer else 0.,checkpoint,pending)
            with self.lock:
                self.snapshot = snap
            last_publish = time.monotonic()
        try:
            # Tiny dense CPU models are faster without a large intra-op thread pool.
            torch.set_num_threads(1)
            if resume:
                trainer,metadata=self.store.resume(resume)
                run,name=metadata['model_identifier'],metadata['model_name']
                if episode_target is not None:
                    from dataclasses import replace
                    if episode_target < trainer.completed:
                        raise ValueError("Episode target must include completed episodes")
                    trainer.config=replace(trainer.config,episodes=episode_target)
                    trainer.environment.config=trainer.config
                checkpoint=str(resume)
            else:
                trainer=Trainer(config or TrainingConfig(),network,architecture)
                checkpoint=str(self.store.save(run,name,trainer))
            # A resumed run gets a separate append-only analysis session. Old data is never rewritten.
            history_run=run+(':session:'+uuid4().hex[:12] if resume else '')
            experiment_id=history_call(experiments.baseline,name,trainer.online,trainer.config,history_run,checkpoint)
            configuration=dict(training=trainer.config.to_dict(),checkpoint_run_id=run,experiment_id=experiment_id,
                               observation_encoder='snake-classic-binary11-v1',resumed_from=str(resume) if resume else None)
            history_call(journal.register,history_run,name,trainer.online.config.sizes,configuration)
            start_steps,start_episodes=trainer.total_steps,trainer.completed
            message='CPU DQN / actual training environment'
            publish()
            accounted_at=time.perf_counter()
            while not shutdown:
                measured_at=time.perf_counter()
                if status=='Running':
                    active_time+=measured_at-accounted_at
                    trainer.elapsed+=measured_at-accounted_at
                accounted_at=measured_at
                while True:
                    try:
                        command,value=self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if command == 'shutdown':
                        shutdown=True
                    elif command in ('pause','stop'):
                        status='Paused' if command=='pause' else 'Stopped'
                        message='Progress retained; resume continues the same episode'
                        history_call(journal.status,history_run,status)
                    elif command=='resume' and trainer.completed < trainer.config.episodes:
                        status,message,deadline='Running','CPU DQN / actual training environment',time.monotonic()
                        history_call(journal.status,history_run,status)
                    elif command=='save':
                        pending=True
                        message='Checkpoint pending: next episode boundary (resume if paused)'
                    elif command=='view':
                        mode,multiplier=value
                        if mode not in ('realtime','accelerated','headless') or multiplier not in (1,2,5,10,20):
                            raise ValueError("Invalid training viewing mode")
                        deadline=time.monotonic()
                    publish()
                if shutdown:
                    break
                if pending and trainer.at_boundary:
                    checkpoint=str(self.store.save(run,name,trainer))
                    if experiment_id:history_call(experiments.link,experiment_id,'checkpoint',checkpoint)
                    pending=False
                    message='Checkpoint saved at episode boundary'
                    publish()
                if trainer.completed >= trainer.config.episodes:
                    if status != 'Completed':
                        checkpoint=str(self.store.save(run,name,trainer))
                        status,message='Completed','Episode target reached; checkpoint saved'
                        history_call(journal.status,history_run,status)
                        if experiment_id:history_call(experiments.link,experiment_id,'checkpoint',checkpoint)
                        publish()
                if status != 'Running':
                    # Event wait is interruptible via the bounded 20 ms poll; no model sharing.
                    threading.Event().wait(.02)
                    continue
                now=time.monotonic()
                if mode!='headless' and now < deadline:
                    threading.Event().wait(min(.02,deadline-now))
                    continue
                started=time.perf_counter()
                transition=trainer.train_step()
                duration=time.perf_counter()-started
                history_call(journal.append_optimization,history_run,trainer.total_steps,trainer.loss,trainer.gradient_norm,trainer.epsilon)
                if transition and transition.done:
                    env=trainer.environment
                    history_call(journal.append_episode,EpisodeRecord(history_run,trainer.completed,env.game.score,env.steps,
                        env.total_reward,transition.reason,trainer.total_steps,trainer.epsilon,trainer.elapsed+duration,
                        env.game.apples_eaten))
                deadline=time.monotonic()+max(0.,1/(9*multiplier)-duration)
                if transition and transition.done and trainer.completed % trainer.config.checkpoint_interval==0:
                    checkpoint=str(self.store.save(run,name,trainer))
                    if experiment_id:history_call(experiments.link,experiment_id,'checkpoint',checkpoint)
                if time.monotonic()-last_publish >= .1 or transition and transition.done:
                    publish()
            status,message='Stopped','Worker shut down at a movement/optimization boundary'
            if history_run and trainer.completed < trainer.config.episodes:history_call(journal.status,history_run,status)
            publish()
        except Exception as error:
            status,message='Failed',str(error)
            if history_run:history_call(journal.status,history_run,status)
            publish()
        finally:
            history_call(journal.close)
