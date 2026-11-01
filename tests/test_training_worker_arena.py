"""Safe training commands and frozen, seeded Arena evaluation."""
import copy
from dataclasses import replace
import time
import pytest
torch=pytest.importorskip('torch',reason='Optional requirements-ai.txt not installed')
from ai.worker import TrainingWorker
from ai.checkpoints import CheckpointStore
from ai.training_config import TrainingConfig
from ai.network import SnakeNetwork
from ai.arena import Arena


def wait_for(worker,predicate,timeout=10):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        snapshot=worker.latest()
        if snapshot and predicate(snapshot):return snapshot
        if snapshot and snapshot.status=='Failed':pytest.fail(snapshot.message)
        time.sleep(.01)
    pytest.fail('Training worker timed out')


def test_worker_start_pause_resume_stop_view_save_and_shutdown(tmp_path):
    worker=TrainingWorker(CheckpointStore(tmp_path))
    try:
        worker.start(TrainingConfig(episodes=10,episode_cap=8,batch_size=2,checkpoint_interval=2))
        wait_for(worker,lambda s:s.values.get('steps',0)>=1)
        worker.command('pause')
        paused=wait_for(worker,lambda s:s.status=='Paused')
        time.sleep(.05)
        assert worker.latest().values['steps']==paused.values['steps']
        worker.command('save')
        wait_for(worker,lambda s:s.pending_save)
        worker.command('view',('headless',1))
        worker.command('resume')
        completed=wait_for(worker,lambda s:s.status=='Completed')
        assert completed.values['completed']==10 and completed.values['steps']<=80
        assert completed.mode=='headless' and completed.checkpoint
        trainer,metadata=worker.store.resume(completed.checkpoint)
        assert trainer.completed==10
        assert completed.steps_per_second>0 and completed.episodes_per_second>0
        assert completed.histories and completed.state is not None
        worker.command('stop')
        stopped=wait_for(worker,lambda s:s.status in ('Stopped','Completed'))
    finally:
        worker.shutdown()
    assert not worker.thread.is_alive()


def test_worker_reports_failure_without_silently_resetting(tmp_path):
    worker=TrainingWorker(CheckpointStore(tmp_path))
    worker.start(resume=tmp_path/'missing'/'checkpoint')
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and worker.latest() is None:time.sleep(.01)
    worker.shutdown()
    assert worker.latest().status=='Failed'


def test_worker_stop_retains_partial_progress_and_shutdown_running_is_safe(tmp_path):
    worker=TrainingWorker(CheckpointStore(tmp_path))
    try:
        worker.start(TrainingConfig(episodes=10,episode_cap=1000))
        wait_for(worker,lambda s:s.values.get('steps',0)>0)
        worker.command('stop')
        stopped=wait_for(worker,lambda s:s.status=='Stopped')
        time.sleep(.05)
        assert worker.latest().values['steps']==stopped.values['steps']
        worker.command('resume')
        wait_for(worker,lambda s:s.status=='Running')
    finally:
        worker.shutdown()
    assert not worker.thread.is_alive() and worker.latest().status=='Stopped'


def greedy_left():
    network=SnakeNetwork()
    with torch.no_grad():
        for p in network.parameters():p.zero_()
        network.layers[-1].bias[1]=1
    return network


def test_arena_greedy_frozen_seeded_replay_and_session_stats():
    arena=Arena(greedy_left(),TrainingConfig(stagnation_factor=2),seed=0,name='Model',episode=50)
    parameters=[p.clone() for p in arena.network.parameters()]
    first=[]
    while not arena.environment.done:
        result=arena.step(); first.append(arena.environment.game.get_state())
        assert result.action==1
    assert arena.games==1 and arena.last_reason=='stagnation'
    assert arena.environment.steps==6
    assert arena.best==0 and arena.total_score==0
    arena.reset()
    second=[]
    while not arena.environment.done:
        arena.step(); second.append(arena.environment.game.get_state())
    assert first==second and arena.games==2
    assert all(torch.equal(a,b) for a,b in zip(parameters,arena.network.parameters()))
    assert all(not p.requires_grad and p.grad is None for p in arena.network.parameters())
    arena.new_game()
    assert arena.seed!=0 and not arena.environment.done


@pytest.mark.parametrize('multiplier',(1,2,5,10))
def test_arena_playback_pause_resume_speed(multiplier):
    arena=Arena(greedy_left())
    arena.multiplier=multiplier
    arena.play()
    for _ in range(120):arena.advance(1/120)
    assert arena.environment.steps==9*multiplier
    state=arena.environment.game.get_state()
    arena.pause();arena.advance(100)
    assert arena.environment.game.get_state()==state
    arena.play();arena.advance(0)
    assert arena.environment.game.get_state()==state


def test_arena_auto_next_delay_and_absolute_limit():
    arena=Arena(greedy_left(),TrainingConfig(episode_cap=2),seed=0)
    arena.auto_next=True
    arena.auto_delay=.5
    arena.play()
    arena.advance(2/9)
    assert arena.environment.done and arena.games==1 and arena.last_reason=='episode_step_limit'
    arena.advance(.4)
    assert arena.environment.done
    arena.advance(.1)
    assert not arena.environment.done and arena.seed!=0 and arena.running
    assert arena.environment.steps==0
