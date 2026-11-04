"""Deterministic shared-learner collection, accounting, recovery and Windows spawn."""
from dataclasses import replace
import json
import time
import pytest
torch=pytest.importorskip('torch')
torch.set_num_threads(1)
from ai.parallel import ParallelTrainer
from ai.frontier_config import FrontierConfig
from ai.training_config import TrainingConfig
from ai.model_registry import model_spec
from ai.environment_pool import EnvironmentPool,episode_seed
from ai.encoders import LEGACY
from ai.checkpoints import CheckpointStore
from ai.frontier import FrontierWorker
from ai.history import HistoryStore
from ai.analysis_repository import AnalysisRepository
from ai.analysis_export import export_named_csv


@pytest.mark.parametrize('count',(1,2,4,8))
@pytest.mark.parametrize('policy',('transition','vector'))
def test_global_accounting_exact_target_and_schedules(count,policy):
    trainer=ParallelTrainer(TrainingConfig(episodes=11,episode_cap=3,batch_size=2,target_sync=1),
        FrontierConfig(environments=count,update_policy=policy))
    events=[]
    try:
        while trainer.completed<11:
            trainer.train_batch();events.extend(trainer.completions)
        assert trainer.total_steps==33 and trainer.completed==11 and len(trainer.replay)==33
        assert [e.index for e in events]==list(range(1,12))
        assert len({(e.environment,e.sequence) for e in events})==11
        assert all(e.length==3 and e.reason=='episode_step_limit' for e in events)
        assert trainer.active_environments==0
        assert trainer.optimizer_updates==(32 if policy=='transition' else trainer.vector_steps-(1 if count==1 else 0))
        assert trainer.epsilon==pytest.approx(1-33*.95/100000)
        assert all(torch.equal(a,b) for a,b in zip(trainer.online.parameters(),trainer.target.parameters()))
    finally:trainer.close()


@pytest.mark.parametrize('architecture','ABCD')
def test_batched_forward_mapping_and_independent_exploration(architecture):
    config=TrainingConfig(episodes=10,batch_size=2,epsilon_start=0.,epsilon_min=0.)
    trainer=ParallelTrainer(config,FrontierConfig(environments=4),architecture=model_spec(architecture))
    try:
        for index in range(4):trainer._start(index)
        calls=[]
        def values(module,args,output):
            calls.append(args[0]);return torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
        hook=trainer.online.register_forward_hook(values)
        assert trainer.choose_actions([3,1,0,2])==[(3,0),(1,1),(0,2),(2,0)]
        hook.remove()
        assert len(calls)==1
        if architecture in 'CD':assert calls[0]['spatial'].shape==(4,3,25,25) and calls[0]['heading'].shape==(4,4)
        else:assert calls[0].shape==(4,11 if architecture=='A' else 19)
        trainer.config=replace(config,epsilon_start=1.,epsilon_min=1.)
        samples=[tuple(action for _,action in trainer.choose_actions([0,1,2,3])) for _ in range(10)]
        assert any(len(set(actions))>1 for actions in samples)
        assert len({s['seed'] for s in trainer.slots})==4
    finally:trainer.close()


@pytest.mark.parametrize('strategy',('vectorized','multiprocessing'))
def test_pool_independent_states_spawn_determinism_and_shutdown(strategy):
    config=TrainingConfig(episode_cap=30)
    pool=EnvironmentPool(config,LEGACY,2,strategy,timeout=10)
    try:
        if strategy=='multiprocessing':assert all(not info['pygame_loaded'] and not info['torch_loaded'] for info in pool.worker_info)
        initial=[pool.reset(i,episode_seed(7,i,0)) for i in range(2)]
        for _ in range(13):result=pool.step([(0,0)])[0]
        assert result[0].done
        other=pool.step([(1,1)])[0]
        assert not other[0].done and other[0].observation==initial[1]
        pool.reset(1,episode_seed(7,1,0))
        assert pool.step([(1,1)])[0]==other
        if strategy=='vectorized':
            assert pool.episodes[0].game is not pool.episodes[1].game
            assert pool.episodes[0].game.apple!=pool.episodes[1].game.apple
    finally:pool.close()
    assert all(not p.is_alive() for p in pool.processes)


@pytest.mark.parametrize('architecture','ABCD')
@pytest.mark.parametrize('policy',('transition','vector'))
def test_partial_checkpoint_exact_resume(architecture,policy,tmp_path):
    execution=FrontierConfig(environments=4,update_policy=policy)
    trainer=ParallelTrainer(TrainingConfig(episodes=25,episode_cap=7,batch_size=2,target_sync=3),execution,architecture=model_spec(architecture,9))
    resumed=None
    try:
        for _ in range(3):trainer.train_batch()
        assert trainer.active_environments==4
        store=CheckpointStore(tmp_path/'checkpoints');path=store.save('parallel','Parallel',trainer)
        resumed,metadata=store.resume(path)
        assert metadata['parallel_execution']['environments']==4
        for _ in range(8):
            assert trainer.train_batch()==resumed.train_batch()
            assert trainer.completions==resumed.completions
        assert trainer.optimizer_updates==resumed.optimizer_updates
        assert all(torch.equal(a,b) for a,b in zip(trainer.online.parameters(),resumed.online.parameters()))
        assert trainer.replay.rng.getstate()==resumed.replay.rng.getstate()
    finally:
        trainer.close()
        if resumed:resumed.close()


def wait(worker,predicate):
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        snapshot=worker.latest()
        if snapshot and predicate(snapshot):return snapshot
        if snapshot and snapshot.status in ('Failed','Save failed'):pytest.fail(snapshot.message)
        time.sleep(.005)
    pytest.fail(str(worker.latest()))


@pytest.mark.parametrize('strategy',('vectorized','multiprocessing'))
def test_worker_pause_save_resume_final_analysis_and_frozen_evaluation(strategy,tmp_path):
    from ai.evaluation import evaluate_checkpoints
    worker=FrontierWorker(tmp_path/'profile',tmp_path/'experiments')
    second=None
    try:
        worker.start(TrainingConfig(episodes=13,episode_cap=6,batch_size=2),FrontierConfig(environments=4,strategy=strategy,headless=False,milestones=(3,7)))
        wait(worker,lambda s:s.status=='Running')
        worker.command('pause');paused=wait(worker,lambda s:s.status=='Paused')
        steps=paused.metrics['steps'];time.sleep(.04)
        assert worker.latest().metrics['steps']==steps
        worker.command('save')
        saved=wait(worker,lambda s:s.checkpoint is not None)
        assert saved.status=='Paused' and saved.metrics['steps']==steps
        worker.command('resume');finished=wait(worker,lambda s:s.status=='Completed')
        assert finished.metrics['completed']==13 and finished.metrics['steps']==78
        assert finished.metrics['environments']==4 and finished.metrics['strategy']==strategy
        root=__import__('pathlib').Path(finished.experiment)
        journal=HistoryStore(root/'history');records=journal.load(root.name).records
        assert len(records)==13 and all(r.environment_id is not None and r.local_sequence is not None for r in records)
        repo=AnalysisRepository(tmp_path/'profile');choice=next(c for c in repo.choices() if str(c.source)==finished.checkpoint)
        dataset=repo.load(choice);assert len(dataset.records)==13
        csv=export_named_csv(tmp_path/'exports',(dataset,));assert 'environment_id' in csv.read_text()
        first=evaluate_checkpoints([__import__('pathlib').Path(finished.checkpoint)],(3,7),tmp_path/'eval')
        again=evaluate_checkpoints([__import__('pathlib').Path(finished.checkpoint)],(3,7),tmp_path/'eval')
        assert json.loads(first.read_text())['models']==json.loads(again.read_text())['models']
        second=FrontierWorker(tmp_path/'profile',tmp_path/'experiments')
        second.start(TrainingConfig(episodes=16),resume=saved.checkpoint)
        extended=wait(second,lambda s:s.status=='Completed')
        assert extended.metrics['completed']==16 and extended.metrics['environments']==4
    finally:
        worker.shutdown()
        if second:second.shutdown()


def test_spawn_worker_failure_is_bounded_and_children_closed():
    import subprocess,sys
    from pathlib import Path
    script="import runpy,sys; runpy.run_path(sys.argv[1],run_name='__mp_main__'); assert not any(name in sys.modules for name in ('app','pygame','torch'))"
    entry=subprocess.run([sys.executable,'-c',script,str(Path(__file__).resolve().parents[1]/'main.py')],capture_output=True,text=True,timeout=10)
    assert entry.returncode==0,entry.stderr
    pool=EnvironmentPool(TrainingConfig(),LEGACY,2,'multiprocessing',timeout=5)
    try:
        pool.processes[0].terminate();pool.processes[0].join(timeout=2)
        with pytest.raises(RuntimeError):pool.step([(0,0)])
    finally:pool.close()
    assert all(not p.is_alive() for p in pool.processes)


def test_duration_eta_and_configuration_validation():
    from services.training_time import duration,CompletionRate
    assert duration(17704)=='4h 55m 04s'
    assert duration(86401)=='1d 00h 00m'
    assert duration(None)=='Calculating…'
    rate=CompletionRate();rate.update(0,0);rate.update(4,10)
    assert rate.remaining(100) is None
    rate.update(5,15);assert rate.remaining(100)>0
    for kwargs in ({'environments':3},{'strategy':'threads'},{'update_policy':'hidden'}):
        with pytest.raises(ValueError):FrontierConfig(**kwargs)


def test_shortened_resume_target_does_not_overshoot_or_count_partial_episodes():
    trainer=ParallelTrainer(TrainingConfig(episodes=20,episode_cap=3,batch_size=2),FrontierConfig(environments=8))
    try:
        trainer.train_batch()
        trainer.config=replace(trainer.config,episodes=1)
        while trainer.completed<1:trainer.train_batch()
        assert trainer.completed==1 and trainer.active_environments==7
        assert sum(trainer.lengths)==3
        trainer.config=replace(trainer.config,episodes=10)
        while trainer.completed<10:trainer.train_batch()
        assert trainer.completed==10 and sum(trainer.lengths)==30
    finally:trainer.close()


def test_stop_releases_processes_and_resume_reconstructs_without_extra_training():
    trainer=ParallelTrainer(TrainingConfig(episodes=10,batch_size=2),FrontierConfig(environments=2,strategy='multiprocessing'))
    try:
        trainer.train_batch()
        state=trainer.snapshot_state();old_processes=trainer.pool.processes
        trainer.close()
        assert all(not p.is_alive() for p in old_processes)
        steps=trainer.total_steps;updates=trainer.optimizer_updates
        trainer.restart_collectors()
        assert trainer.snapshot_state()==state and trainer.total_steps==steps and trainer.optimizer_updates==updates
        trainer.train_batch();assert trainer.total_steps==steps+2
    finally:trainer.close()


def test_worker_failure_records_recovery_without_deadlock(tmp_path,monkeypatch):
    original=ParallelTrainer.train_batch
    calls=[]
    def fail(self,*args,**kwargs):
        calls.append(1)
        if len(calls)==3:raise RuntimeError('Injected collector failure')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(ParallelTrainer,'train_batch',fail)
    worker=FrontierWorker(tmp_path/'profile',tmp_path/'experiments')
    restored=None
    try:
        worker.start(TrainingConfig(episodes=20,batch_size=2),FrontierConfig(environments=4))
        failed=wait(worker,lambda s:s.status=='Failed')
        assert 'Injected collector failure' in failed.message and failed.checkpoint
        restored,_=CheckpointStore(tmp_path/'profile/checkpoints').resume(failed.checkpoint)
        assert restored.total_steps==8
    finally:
        worker.shutdown()
        if restored:restored.close()


def test_parallel_ui_controls_and_completed_time(tmp_path):
    import os
    os.environ.update(SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy')
    from app import Application
    from ai.laboratory import Laboratory
    from ui.neural_view import NeuralView
    app=Application(tmp_path/'settings.json')
    app.lab_view=NeuralView(Laboratory(),tmp_path/'models')
    try:
        app.action('open_frontier');view=app.frontier_view
        app.action('frontier:environments');app.action('frontier:environments')
        assert view.environments==4
        app.action('frontier:update_policy');assert view.update_policy=='vector'
        assert view.configuration()[2].resolved_strategy=='vectorized'
        view.text.update(episodes='7',episode_cap='3',batch_size='2')
        app.draw(.016);app.action('frontier:start')
        completed=wait(view.worker,lambda s:s.status=='Completed')
        app.update(.016);app.draw(.016)
        assert completed.metrics['environments']==4 and completed.metrics['update_policy']=='vector'
        assert completed.metrics['active_environments']==0 and completed.metrics['completed']==7
    finally:app.close()


def test_historical_checkpoint_without_new_optimizer_field_resumes_identically(tmp_path):
    import hashlib
    from ai.dqn import Trainer
    trainer=Trainer(TrainingConfig(episodes=10,episode_cap=3,batch_size=2))
    for _ in range(6):trainer.train_step()
    store=CheckpointStore(tmp_path/'checkpoints');path=store.save('legacy','Old',trainer)
    state=torch.load(path/'training.pt',weights_only=True);state.pop('optimizer_updates')
    torch.save(state,path/'training.pt')
    metadata=json.loads((path/'training.json').read_text());metadata.pop('optimizer_updates')
    metadata['sha256']=hashlib.sha256((path/'training.pt').read_bytes()).hexdigest()
    (path/'training.json').write_text(json.dumps(metadata))
    restored,_=store.resume(path)
    assert restored.optimizer_updates==5
    for _ in range(6):assert restored.train_step()==trainer.train_step()
    assert all(torch.equal(a,b) for a,b in zip(restored.online.parameters(),trainer.online.parameters()))
