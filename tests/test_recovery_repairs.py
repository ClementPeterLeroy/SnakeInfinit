from pathlib import Path
import time
import pytest
torch=pytest.importorskip('torch')
from ai.parallel import ParallelTrainer
from ai.training_config import TrainingConfig
from ai.frontier_config import FrontierConfig
from ai.frontier import FrontierWorker,FrontierHistory
from ai.checkpoints import CheckpointStore
from ai.history import HistoryStore
from ai.analysis_stats import EpisodeRecord


@pytest.mark.parametrize('phase',('before_collection','partial_collection','before_optimizer','after_optimizer','terminal_cleanup'))
def test_interrupted_batch_is_never_resumable(phase,tmp_path,monkeypatch):
    trainer=ParallelTrainer(TrainingConfig(episodes=20,episode_cap=1,batch_size=2),FrontierConfig(environments=4))
    try:
        def failure(*args,**kwargs):raise RuntimeError('injected '+phase)
        if phase=='before_collection':monkeypatch.setattr(trainer,'choose_actions',failure)
        elif phase=='partial_collection':
            step=trainer.pool.step
            def partial(actions):step(actions[:1]);failure()
            monkeypatch.setattr(trainer.pool,'step',partial)
        elif phase=='before_optimizer':monkeypatch.setattr(trainer,'optimize',failure)
        elif phase=='after_optimizer':
            def mutate():
                with torch.no_grad():next(trainer.online.parameters()).add_(1)
                failure()
            monkeypatch.setattr(trainer,'optimize',mutate)
        else:
            class BrokenList(list):
                def append(self,value):failure()
            trainer.completions=BrokenList()
            # train_batch creates a fresh list; inject failure at terminal score-history cleanup.
            trainer.scores=BrokenList()
        with pytest.raises(RuntimeError):trainer.train_batch()
        assert not trainer.recovery_safe and not trainer.at_boundary
        for _ in range(2):
            with pytest.raises(ValueError):CheckpointStore(tmp_path/'cp').save('run','Run',trainer)
        assert not (tmp_path/'cp').exists()
    finally:trainer.close()


def test_previous_recovery_survives_failed_next_batch(tmp_path,monkeypatch):
    original=ParallelTrainer.optimize;calls=[]
    def fail(self):
        calls.append(1)
        if len(calls)==6:raise RuntimeError('partial optimizer failure')
        return original(self)
    monkeypatch.setattr(ParallelTrainer,'optimize',fail)
    worker=FrontierWorker(tmp_path/'profile',tmp_path/'experiments');restored=None
    try:
        worker.start(TrainingConfig(episodes=20,episode_cap=1,batch_size=2),FrontierConfig(environments=4,recovery_interval=4,milestones=()))
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            snapshot=worker.latest()
            if snapshot and snapshot.status=='Failed':break
            time.sleep(.005)
        assert snapshot.status=='Failed' and 'previous verified recovery' in snapshot.message
        restored,_=CheckpointStore(tmp_path/'profile/checkpoints').resume(snapshot.checkpoint)
        assert restored.completed==4 and snapshot.metrics['completed']==5
        worker.shutdown()
        path=Path(snapshot.experiment)
        assert len(HistoryStore(path/'history').load(path.name).records)==5
    finally:
        worker.shutdown()
        if restored:restored.close()


def test_completion_retry_is_idempotent_and_flush_failure_keeps_queue(tmp_path,monkeypatch):
    journal=FrontierHistory(tmp_path)
    record=EpisodeRecord('run',1,0,length=1,seed=7)
    diagnostics=('run',1,0,None,None,None)
    try:
        journal.register('run','Run',(11,16,8,3),{})
        journal.record(record,diagnostics);journal.record(record,diagnostics)
        original=journal._writer
        def fail():raise OSError('temporary disk error')
        monkeypatch.setattr(journal,'_writer',fail)
        with pytest.raises(OSError):journal.flush()
        assert len(journal.pending)==1
        monkeypatch.setattr(journal,'_writer',original)
        journal.flush();journal.record(record,diagnostics);journal.flush()
        assert len(journal.load('run').records)==1
    finally:journal.close()
