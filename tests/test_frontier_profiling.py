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
