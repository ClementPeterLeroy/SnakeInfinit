"""One-environment Frontier execution, artifacts, recording and exact boundary resume."""
import json
from pathlib import Path
import sqlite3
import time
import pytest
torch=pytest.importorskip('torch')
torch.set_num_threads(1)
from ai.dqn import Trainer
from ai.frontier import FrontierWorker
from ai.frontier_config import FrontierConfig
from ai.frontier_store import FrontierStore,load_published_model
from ai.checkpoints import CheckpointStore
from ai.training_config import TrainingConfig
from ai.analysis_repository import AnalysisRepository
from ai.analysis_export import export_csv
from ai.history import HistoryStore
from ai.arena import Arena


def wait(worker,predicate,timeout=20,ignore=None):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        snap=worker.latest()
        if snap is ignore:
            time.sleep(.005)
            continue
        if snap and predicate(snap):return snap
        if snap and (snap.status in ('Failed','Save failed') or snap.status=='Paused' and 'failed' in snap.message):pytest.fail(snap.message)
        time.sleep(.005)
    pytest.fail('Frontier timeout: '+str(worker.latest()))


@pytest.mark.parametrize('kwargs',[{'refresh_episodes':1},{'recovery_interval':0},{'recovery_slots':0},
    {'milestones':(0,)},{'headless':'yes'},{'profiling':1}])
def test_frontier_configuration_validation(kwargs):
    with pytest.raises(ValueError):FrontierConfig(**kwargs)


def test_frontier_defaults():
    policy=FrontierConfig()
    assert policy.headless and policy.refresh_episodes==100 and policy.milestones==(1000,3000,5000,10000)
    assert policy.recovery_slots==2 and TrainingConfig().episodes==10000


def test_headless_records_snapshots_recovery_final_arena_and_analysis(tmp_path):
    worker=FrontierWorker(tmp_path/'profile',tmp_path/'experiments')
    try:
        worker.start(TrainingConfig(episodes=6,episode_cap=3,batch_size=2),
            FrontierConfig(recovery_interval=2,milestones=(1,3,5)))
        result=wait(worker,lambda s:s.status=='Completed')
        assert result.metrics['completed']==6 and result.metrics['steps']==18
        assert result.publications<=4  # Initial + final, not per-step/per-episode.
        root=Path(result.experiment)
        assert len(list((root/'models').glob('*_model.pt')))==4
        assert result.model and result.model.endswith('ep000006_model.pt')
        journal=HistoryStore(root/'history')
        run=journal.datasets()[0]['identifier']
        records=journal.load(run).records
        assert len(records)==6 and all(r.seed is not None and r.reason=='episode_step_limit' for r in records)
        assert all(r.length==3 and r.cumulative_steps==r.episode*3 for r in records)
        with sqlite3.connect(journal.path) as db:
            assert db.execute('SELECT COUNT(*) FROM episode_diagnostics').fetchone()[0]==6
            assert db.execute('SELECT SUM(optimizer_updates) FROM episode_diagnostics').fetchone()[0]==17
        canonical=CheckpointStore(tmp_path/'profile/checkpoints')
        entries=canonical.entries()
        assert len([e for e in entries if e['checkpoint'].startswith('recovery-')])==2
        model,metadata=load_published_model(result.model)
        assert model(torch.zeros(11)).shape==(3,)
        arena=Arena(model,TrainingConfig(episode_cap=3),seed=12)
        first=[]
        while not arena.environment.done:
            arena.step();first.append(arena.environment.game.get_state())
        arena.reset(12)
        second=[]
        while not arena.environment.done:
            arena.step();second.append(arena.environment.game.get_state())
        assert arena.games==2 and first==second
        repo=AnalysisRepository(tmp_path/'profile')
        choices=repo.choices()
        selected=next(c for c in choices if str(c.source)==result.checkpoint)
        data=repo.load(selected)
        assert len(data.records)==6 and data.checkpoint_episode==6
        assert export_csv(tmp_path/'raw.csv',(data,)).is_file()
        resumed,_=canonical.resume(result.checkpoint)
        assert resumed.completed==6 and resumed.total_steps==18
        from ai.evaluation import evaluate_checkpoints
        evaluation=evaluate_checkpoints([Path(result.checkpoint)],(12,13),tmp_path/'profile/evaluations')
        payload=json.loads(evaluation.read_text())
        assert payload['seeds']==[12,13] and len(payload['models'][0]['records'])==2
    finally:worker.shutdown()
    assert not worker.thread.is_alive()


def test_boundary_resume_is_numerically_identical_and_new_experiment_isolated(tmp_path):
    config=TrainingConfig(episodes=10,episode_cap=3,batch_size=2)
    trainer=Trainer(config)
    for _ in range(9):trainer.train_step()
    store=CheckpointStore(tmp_path/'profile/checkpoints')
    path=store.save('baseline','Baseline',trainer)
    before={p.name:p.read_bytes() for p in path.iterdir()}
    resumed,_=store.resume(path)
    for _ in range(9):
        assert trainer.train_step()==resumed.train_step()
    assert all(torch.equal(a,b) for a,b in zip(trainer.online.parameters(),resumed.online.parameters()))
    worker=FrontierWorker(tmp_path/'profile',tmp_path/'experiments')
    try:
        worker.start(config,resume=path)
        result=wait(worker,lambda s:s.status=='Completed')
        assert result.metrics['completed']==10
        while trainer.completed<10:trainer.train_step()
        final,_=store.load_model(result.checkpoint)
        assert all(torch.equal(a,b) for a,b in zip(trainer.online.parameters(),final.parameters()))
        assert before=={p.name:p.read_bytes() for p in path.iterdir()}
        assert Path(result.experiment).name!='baseline'
    finally:worker.shutdown()


def test_pause_resume_preserves_partial_episode_and_stop_save_boundary(tmp_path):
    worker=FrontierWorker(tmp_path/'profile',tmp_path/'experiments')
    try:
        worker.start(TrainingConfig(episodes=100,episode_cap=100,batch_size=2),FrontierConfig(headless=False))
        wait(worker,lambda s:s.metrics.get('steps',0)>=0)
        worker.command('pause')
        paused=wait(worker,lambda s:s.status=='Paused')
        steps=paused.metrics['steps'];time.sleep(.03)
        assert worker.latest().metrics['steps']==steps
        worker.command('save_stop')
        saved=wait(worker,lambda s:s.status=='Stopped' and s.checkpoint is not None)
        trainer,_=CheckpointStore(tmp_path/'profile/checkpoints').resume(saved.checkpoint)
        assert trainer.at_boundary and trainer.total_steps>=steps
        worker.command('resume')
        wait(worker,lambda s:s.status=='Running')
    finally:worker.shutdown()
    assert worker.latest().status=='Stopped'


def test_atomic_visibility_and_experiment_directory_collision(tmp_path,monkeypatch):
    config=TrainingConfig(episodes=1,episode_cap=1)
    store=FrontierStore(tmp_path/'experiments',tmp_path/'profile',config,FrontierConfig(),'Test',identifier='unique')
    try:
        with pytest.raises(FileExistsError):FrontierStore(tmp_path/'experiments',tmp_path/'profile',config,FrontierConfig(),'Test',identifier='unique')
        trainer=Trainer(config);trainer.train_step()
        folder,_=store.save(trainer)
        pointer=(store.root/'recovery.json').read_bytes()
        import ai.checkpoints
        original=torch.save
        def fail(state,path,*args,**kwargs):
            if Path(path).name=='training.pt':raise OSError('simulated interrupted save')
            return original(state,path,*args,**kwargs)
        monkeypatch.setattr(torch,'save',fail)
        with pytest.raises(OSError):store.save(trainer)
        assert (store.root/'recovery.json').read_bytes()==pointer
        assert [entry['path'] for entry in store.checkpoints.entries()]==[folder]
        assert store.checkpoints.resume(folder)[0].completed==1
    finally:store.close()


def test_real_default_milestone_1000_and_nonstandard_final_budget(tmp_path):
    worker=FrontierWorker(tmp_path/'profile',tmp_path/'experiments')
    try:
        worker.start(TrainingConfig(episodes=1001,episode_cap=1,batch_size=4096),
            FrontierConfig(recovery_interval=10000,refresh_episodes=500))
        result=wait(worker,lambda s:s.status=='Completed')
        files=list((Path(result.experiment)/'models').glob('*_model.pt'))
        assert {p.name for p in files}=={'snake_dqn_11_ep001000_model.pt','snake_dqn_11_ep001001_model.pt'}
        assert result.metrics['optimizer_updates']==0
        assert result.publications==4  # Initial, episode 500, episode 1000, final.
        entries=CheckpointStore(tmp_path/'profile/checkpoints').entries()
        milestone=next(e for e in entries if e['episode']==1000)
        with pytest.raises(ValueError,match='Model-only'):CheckpointStore(tmp_path/'profile/checkpoints').resume(milestone['path'])
    finally:worker.shutdown()


def test_failed_final_save_is_visible_and_can_be_retried(tmp_path,monkeypatch):
    original=FrontierStore.publish_model
    calls=[]
    def fail_once(self,*args):
        calls.append(1)
        if len(calls)==1:raise OSError('fixture save failure')
        return original(self,*args)
    monkeypatch.setattr(FrontierStore,'publish_model',fail_once)
    worker=FrontierWorker(tmp_path/'profile',tmp_path/'experiments')
    try:
        worker.start(TrainingConfig(episodes=1,episode_cap=1))
        failed=wait(worker,lambda s:s.status=='Save failed')
        assert failed.model is None and 'fixture save failure' in failed.message
        worker.command('save')
        saved=wait(worker,lambda s:s.status=='Completed',ignore=failed)
        assert Path(saved.model).is_file() and saved.metrics['completed']==1
    finally:worker.shutdown()
