"""Real CPU DQN math, bounded episodes and exact resumption."""
from dataclasses import replace
import json
import subprocess
import sys
import pytest
torch=pytest.importorskip('torch',reason='Optional requirements-ai.txt not installed')
torch.set_num_threads(1)
from ai.training_config import TrainingConfig
from ai.config import NetworkConfig
from ai.environment import Episode,Transition
from ai.replay import ReplayBuffer
from ai.dqn import Trainer,bellman_targets
from ai.checkpoints import CheckpointStore
from core.game import Game


def transition(reward=1.,terminated=False,truncated=False):
    return Transition((1.,)*11,1,reward,(0.,)*11,terminated,truncated,
                      'stagnation' if truncated else 'wall_collision' if terminated else None)


@pytest.mark.parametrize('kwargs',[{'episodes':0},{'learning_rate':float('nan')},{'gamma':1.1},
    {'batch_size':100,'replay_capacity':10},{'epsilon_min':.8,'epsilon_start':.2},
    {'target_sync':0},{'checkpoint_interval':0},{'seed':-1},{'episode_cap':0},{'stagnation_factor':0}])
def test_invalid_training_configs_rejected(kwargs):
    with pytest.raises(ValueError): TrainingConfig(**kwargs)


def test_replay_is_bounded_and_samples_reproducibly():
    a,b=ReplayBuffer(3,4),ReplayBuffer(3,4)
    for i in range(6):
        a.append(transition(i)); b.append(transition(i))
    assert len(a)==3 and sorted(t.reward for t in a.items)==[3,4,5]
    assert a.sample(3)==b.sample(3)
    assert len(set(t.reward for t in a.sample(3)))==3
    with pytest.raises(ValueError): a.sample(4)


def test_bellman_targets_mask_termination_and_truncation():
    targets=bellman_targets(torch.tensor([1.,-10.,-5.]),torch.tensor([100.,100.,100.]),
                            torch.tensor([False,True,True]),.9)
    assert targets.tolist()==[91.,-10.,-5.]


def test_epsilon_streams_are_owned_reproducible_and_greedy_at_zero():
    config=TrainingConfig(epsilon_start=1,epsilon_min=1)
    first,second=Trainer(config),Trainer(config)
    a=[first.choose_action((0.,)*11) for _ in range(100)]
    b=[second.choose_action((0.,)*11) for _ in range(100)]
    assert a==b and set(a)=={0,1,2}
    first.config=replace(config,epsilon_start=0,epsilon_min=0)
    with torch.no_grad():
        for p in first.online.parameters():p.zero_()
        first.online.layers[-1].bias[2]=5
    assert {first.choose_action((0.,)*11) for _ in range(20)}=={2}
    first.config=TrainingConfig(epsilon_decay_steps=100)
    first.total_steps=50
    assert first.epsilon==pytest.approx(.525)
    first.total_steps=1000
    assert first.epsilon==pytest.approx(.05)


def test_optimizer_warmup_huber_clipping_and_target_independence():
    trainer=Trainer(TrainingConfig(batch_size=4,gradient_clip=.0001))
    assert trainer.online(torch.zeros(11)).shape==(3,)
    target=[p.clone() for p in trainer.target.parameters()]
    online=[p.clone() for p in trainer.online.parameters()]
    assert trainer.optimize() is None
    for _ in range(4):trainer.replay.append(transition(1000))
    assert trainer.optimize()>900 and trainer.gradient_norm>trainer.config.gradient_clip
    norm=torch.sqrt(sum((p.grad**2).sum() for p in trainer.online.parameters()))
    assert norm<=trainer.config.gradient_clip+1e-6
    assert any(not torch.equal(a,b) for a,b in zip(online,trainer.online.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(target,trainer.target.parameters()))
    assert all(p.grad is None and not p.requires_grad for p in trainer.target.parameters())


def test_target_sync_at_environment_step_interval():
    trainer=Trainer(TrainingConfig(target_sync=2,batch_size=1))
    trainer.train_step()
    assert any(not torch.equal(a,b) for a,b in zip(trainer.online.parameters(),trainer.target.parameters()))
    trainer.train_step()
    assert all(torch.equal(a,b) for a,b in zip(trainer.online.parameters(),trainer.target.parameters()))


def test_rewards_food_collision_stagnation_cap_and_reset():
    env=Episode()
    env.game.apple=(13,12)
    result=env.step(0)
    assert result.reward==10 and env.game.score==1 and env.without_food==0
    assert env.step(0).reward==-.01
    env.game.snake.body=[(24,5),(23,5),(22,5)]
    result=env.step(0)
    assert result.reward==-10 and result.terminated and result.reason=='wall_collision'
    assert env.reset(0)==Episode(seed=0).observation() and env.steps==0 and not env.done
    env=Episode(TrainingConfig(stagnation_factor=1))
    for _ in range(3): result=env.step(1)
    assert result.truncated and not result.terminated and result.reason=='stagnation' and result.reward==-5
    env=Episode(TrainingConfig(episode_cap=2))
    env.step(1); result=env.step(1)
    assert result.truncated and result.reason=='episode_step_limit'
    with pytest.raises(ValueError):env.step(1)


def test_self_collision_and_board_completion_are_distinct():
    env=Episode()
    env.game.snake.body=[(5,5),(4,5),(4,4),(5,4),(6,4),(6,5),(6,6)]
    assert env.step(0).reason=='self_collision'
    env=Episode()
    env.game=Game(size=3)
    env.game.snake.body=[(1,0),(0,0),(0,1),(1,1),(2,1),(2,2),(1,2),(0,2)]
    env.game.apple=(2,0)
    result=env.step(0)
    assert result.terminated and not result.truncated and result.reason=='board_completion'


def test_nonfinite_values_fail_without_reinitializing_model():
    trainer=Trainer(TrainingConfig(epsilon_start=0,epsilon_min=0,batch_size=1))
    with torch.no_grad():trainer.online.layers[-1].bias[0]=float('nan')
    with pytest.raises(ValueError,match='Non-finite'):trainer.choose_action((0.,)*11)
    assert torch.isnan(trainer.online.layers[-1].bias[0])
    trainer=Trainer(TrainingConfig(batch_size=1))
    trainer.replay.append(transition(float('inf')))
    with pytest.raises(ValueError,match='Non-finite'):trainer.optimize()


def test_checkpoint_exact_resume_ring_buffer_optimizer_target_and_rng(tmp_path):
    config=TrainingConfig(episodes=8,episode_cap=20,replay_capacity=40,batch_size=4,target_sync=7)
    trainer=Trainer(config,architecture=NetworkConfig((12,6),'tanh',5))
    while trainer.completed<3:trainer.train_step()
    store=CheckpointStore(tmp_path)
    folder=store.save('run','Test Model',trainer,checkpoint='episode-3')
    with pytest.raises(FileExistsError):store.save('run','Test Model',trainer,checkpoint='episode-3')
    restored,metadata=store.resume(folder)
    assert restored.online.config==trainer.online.config
    assert restored.epsilon==trainer.epsilon and restored.completed==3
    assert restored.replay.items==trainer.replay.items and restored.replay.position==trainer.replay.position
    assert restored.optimizer.state_dict()['param_groups']==trainer.optimizer.state_dict()['param_groups']
    while trainer.completed<8:trainer.train_step()
    while restored.completed<8:restored.train_step()
    assert trainer.total_steps==restored.total_steps and trainer.metrics()==restored.metrics()
    assert all(torch.equal(a,b) for a,b in zip(trainer.online.parameters(),restored.online.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(trainer.target.parameters(),restored.target.parameters()))
    assert list(trainer.scores)==list(restored.scores)
    assert store.entries()[0]['name']=='Test Model'


@pytest.mark.parametrize('kind',('json','weights','resume','configuration'))
def test_checkpoint_corruption_rejected(tmp_path,kind):
    store=CheckpointStore(tmp_path)
    folder=store.save('run','Model',Trainer())
    if kind=='json':(folder/'training.json').write_text('{broken')
    elif kind=='resume':(folder/'training.pt').write_bytes(b'corrupted')
    elif kind=='weights':
        path=next(folder.glob('weights*.pt')); path.write_bytes(b'corrupted')
    else:
        path=folder/'training.json'; data=json.loads(path.read_text())
        data['hyperparameters']['gamma']=4; path.write_text(json.dumps(data))
    with pytest.raises(ValueError):store.resume(folder)


def test_training_imports_no_pygame_and_bounds_all_episodes():
    script='''
import builtins
old=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0]=='pygame': raise AssertionError(name)
    return old(name,*args,**kwargs)
builtins.__import__=guard
import torch
torch.set_num_threads(1)
from ai.dqn import Trainer
from ai.training_config import TrainingConfig
t=Trainer(TrainingConfig(episodes=3,episode_cap=5,batch_size=2))
while t.completed<3:t.train_step()
assert t.total_steps<=15
'''
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
