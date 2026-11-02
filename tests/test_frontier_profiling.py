"""Optional timing must not change the original DQN's numerical behavior."""
import pytest
torch=pytest.importorskip('torch')
from ai.dqn import Trainer
from ai.profiling import Profiler,process_usage
from ai.training_config import TrainingConfig


def test_profiling_preserves_trajectories_weights_and_rngs():
    config=TrainingConfig(episodes=20,episode_cap=4,batch_size=2)
    baseline,profiled=Trainer(config),Trainer(config)
    timer=Profiler(True);timer.attach(profiled)
    for _ in range(40):assert baseline.train_step()==profiled.train_step()
    assert all(torch.equal(a,b) for a,b in zip(baseline.online.parameters(),profiled.online.parameters()))
    assert baseline.action_rng.getstate()==profiled.action_rng.getstate()
    assert baseline.replay.rng.getstate()==profiled.replay.rng.getstate()
    result=timer.report()
    assert result['environment_step']['calls']==40 and result['replay_sampling']['calls']==39
    assert result['optimizer_update']['calls']==39
    assert all(value['seconds']>=0 for value in result.values())
    disabled=Profiler(False);disabled.attach(baseline)
    assert disabled.seconds=={}


def test_process_measurements_are_labeled_and_optional():
    result=process_usage()
    assert result['cpu_seconds']>=0
    assert result['ram_bytes'] is None or result['ram_bytes']>0
    assert result['peak_ram_bytes'] is None or result['peak_ram_bytes']>0


def test_inference_timer_counts_forward_calls_only():
    trainer=Trainer(TrainingConfig(episodes=10,batch_size=2,episode_cap=4,epsilon_start=0.,epsilon_min=0.))
    timer=Profiler(True);timer.attach(trainer)
    for _ in range(8):trainer.train_step()
    result=timer.report()
    assert result['neural_inference']['calls']==result['action_selection']['calls']==8
    # Optimization also runs online.forward; those calls must not count as policy inference.
    assert result['optimizer_update']['calls']==7
    exploratory=Trainer(TrainingConfig(epsilon_start=1.,epsilon_min=1.))
    random_timer=Profiler(True);random_timer.attach(exploratory)
    exploratory.train_step()
    assert random_timer.report()['neural_inference']['calls']==0
