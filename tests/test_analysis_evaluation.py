"""Legacy compatibility, fair frozen evaluations and failure-isolated journals."""
import hashlib
import json
import sqlite3
import threading
import time
import pytest

torch=pytest.importorskip('torch')
from ai.analysis_model import AnalysisModel
from ai.analysis_repository import AnalysisRepository
from ai.checkpoints import CheckpointStore
from ai.config import NetworkConfig
from ai.dqn import Trainer
from ai.evaluation import evaluate_checkpoints
from ai.history import HistoryStore,legacy_dataset
from ai.training_config import TrainingConfig
from ai.worker import TrainingWorker


def snapshot(store,run='baseline',architecture=None,episodes=3):
    trainer=Trainer(TrainingConfig(episodes=episodes,episode_cap=5,batch_size=2),architecture=architecture)
    while trainer.completed<episodes:trainer.train_step()
    return store.save(run,run,trainer)


def hashes(folder):
    return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}


def test_legacy_history_read_only_without_constructing_model(tmp_path,monkeypatch):
    store=CheckpointStore(tmp_path/'checkpoints')
    path=snapshot(store)
    original=hashes(tmp_path)
    monkeypatch.setattr(CheckpointStore,'load_model',lambda *a:pytest.fail('History must not construct a neural model'))
    data=legacy_dataset(path)
    assert len(data.records)==3 and data.architecture==(11,16,8,3)
    assert data.records[0].cumulative_steps is None and data.records[-1].cumulative_steps==15
    assert data.records[0].reason is None and data.records[-1].reason=='episode_step_limit'
    assert data.limitations and hashes(tmp_path)==original


def test_same_checkpoint_two_architectures_frozen_evaluation_and_repository(tmp_path):
    store=CheckpointStore(tmp_path/'checkpoints')
    paths=[snapshot(store),snapshot(store,'wide',NetworkConfig(hidden_sizes=(32,)))]
    before=hashes(tmp_path/'checkpoints')
    config=TrainingConfig(episode_cap=7,stagnation_factor=2).to_dict()
    first=evaluate_checkpoints(paths,(4,8,12),tmp_path/'evaluations',configuration=config)
    second=evaluate_checkpoints(paths,(4,8,12),tmp_path/'evaluations',configuration=config)
    a,b=json.loads(first.read_text()),json.loads(second.read_text())
    assert a['models']==b['models'] and a['seeds']==[4,8,12]
    assert a['configuration']['episode_cap']==7
    assert all(row['epsilon']==0 and row['length']<=7 for model in a['models'] for row in model['records'])
    assert hashes(tmp_path/'checkpoints')==before
    repo=AnalysisRepository(tmp_path)
    choices=[c for c in repo.choices() if c.kind=='evaluation' and c.source==first]
    datasets=tuple(repo.load(c) for c in choices)
    assert len(datasets)==2 and datasets[0].architecture!=datasets[1].architecture
    assert AnalysisModel().prepare(datasets)['models'][0]['summary']['count']==3
    other=evaluate_checkpoints(paths[:1],(5,8,12),tmp_path/'evaluations',configuration=config)
    choice=next(c for c in repo.choices() if c.source==other)
    with pytest.raises(ValueError,match='identical recorded seed'):
        AnalysisModel().prepare((datasets[0],repo.load(choice)))


def test_evaluation_cancellation_leaves_no_partial_result(tmp_path):
    path=snapshot(CheckpointStore(tmp_path/'checkpoints'))
    cancel=threading.Event();cancel.set()
    with pytest.raises(InterruptedError,match='canceled'):
        evaluate_checkpoints([path],(0,1),tmp_path/'evaluations',cancel)
    assert not (tmp_path/'evaluations').exists()


def complete(worker):
    worker.command('view',('headless',1))
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        state=worker.latest()
        if state and state.status in ('Completed','Failed'):return state
        time.sleep(.01)
    pytest.fail('Worker timed out')


def test_worker_durable_raw_journal_and_resume_branch(tmp_path):
    worker=TrainingWorker(CheckpointStore(tmp_path/'checkpoints'))
    try:
        worker.start(TrainingConfig(episodes=3,episode_cap=5,batch_size=2))
        result=complete(worker)
        assert result.status=='Completed'
    finally:worker.shutdown()
    journal=HistoryStore(tmp_path/'history')
    run=journal.datasets()[0]
    data=journal.load(run['identifier'])
    assert run['status']=='Completed' and len(data.records)==3
    assert all(r.length==5 and r.reason=='episode_step_limit' and r.epsilon is not None and r.elapsed>=0 for r in data.records)
    with sqlite3.connect(journal.path) as db:
        assert db.execute('SELECT COUNT(*) FROM optimization').fetchone()[0]==15
    original=data.records
    resumed=TrainingWorker(worker.store)
    try:
        resumed.start(resume=result.checkpoint,episode_target=4)
        assert complete(resumed).status=='Completed'
    finally:resumed.shutdown()
    assert journal.load(run['identifier']).records==original
    branch=journal.datasets()[1]
    assert ':session:' in branch['identifier']
    assert [r.episode for r in journal.load(branch['identifier']).records]==[4]


def test_history_write_failure_does_not_stop_training(tmp_path,monkeypatch):
    def fail(*args):raise OSError('fixture disk failure')
    monkeypatch.setattr(HistoryStore,'append_episode',fail)
    worker=TrainingWorker(CheckpointStore(tmp_path/'checkpoints'))
    try:
        worker.start(TrainingConfig(episodes=3,episode_cap=2))
        result=complete(worker)
        assert result.status=='Completed' and result.values['completed']==3
        assert 'History warning' in result.message
    finally:worker.shutdown()
