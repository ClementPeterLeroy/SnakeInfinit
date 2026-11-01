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
        def publish():
            nonlocal last_publish
            metrics = trainer.metrics() if trainer else {}
            seconds = max(active_time,1e-9)
            snap = TrainingSnapshot(status,message,tuple(metrics.items()),
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
                    elif command=='resume' and trainer.completed < trainer.config.episodes:
                        status,message,deadline='Running','CPU DQN / actual training environment',time.monotonic()
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
                    pending=False
                    message='Checkpoint saved at episode boundary'
                    publish()
                if trainer.completed >= trainer.config.episodes:
                    if status != 'Completed':
                        checkpoint=str(self.store.save(run,name,trainer))
                        status,message='Completed','Episode target reached; checkpoint saved'
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
                deadline=time.monotonic()+max(0.,1/(9*multiplier)-duration)
                if transition and transition.done and trainer.completed % trainer.config.checkpoint_interval==0:
                    checkpoint=str(self.store.save(run,name,trainer))
                if time.monotonic()-last_publish >= .1 or transition and transition.done:
                    publish()
            status,message='Stopped','Worker shut down at a movement/optimization boundary'
            publish()
        except Exception as error:
            status,message='Failed',str(error)
            publish()
