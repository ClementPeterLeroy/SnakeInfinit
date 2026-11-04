"""Audit-only contract checks and strict expected failures; no production repairs."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import pytest
torch=pytest.importorskip('torch')
torch.set_num_threads(1)
from ai.environment import Episode
from ai.environment_pool import EnvironmentPool,episode_seed
from ai.encoders import LEGACY,DISTANCE,BINARY,get_encoder
from ai.parallel import ParallelTrainer
from ai.frontier_config import FrontierConfig
from ai.training_config import TrainingConfig
from ai.checkpoints import CheckpointStore
from ai.frontier import FrontierWorker
from ai.history import HistoryStore
from core.types import Direction


@pytest.mark.parametrize('count',(1,2,4,8))
def test_audit_local_state_rng_score_reward_and_reset_independence(count):
    config=TrainingConfig(episodes=100,stagnation_factor=1)
    pool=EnvironmentPool(config,LEGACY,count)
    try:
        for i in range(count):pool.reset(i,episode_seed(17,i,0))
        games=[episode.game for episode in pool.episodes]
        for values in (games,[g.board for g in games],[g.snake for g in games],[g.snake.body for g in games],[g.rng for g in games]):
            assert len({id(v) for v in values})==count
        initial=[g.get_state() for g in games];rngs=[g.rng.getstate() for g in games]
        games[0].apple=(13,12)
        transition,score,apples=pool.step([(0,0)])[0]
        assert score==apples==1 and transition.reward==config.food_reward
        assert games[0].rng.getstate()!=rngs[0]
        assert all(g.get_state()==state and g.rng.getstate()==rng for g,state,rng in zip(games[1:],initial[1:],rngs[1:]))
        pool.reset(0,episode_seed(17,0,1))
        assert games[0].score==0
        games[0].snake.body=[(24,12),(23,12),(22,12)];games[0].snake.direction=Direction.RIGHT
        collision=pool.step([(0,0)])[0][0]
        assert collision.reason=='wall_collision' and collision.reward==config.collision_reward
        assert all(g.get_state()==state for g,state in zip(games[1:],initial[1:]))
        pool.reset(0,episode_seed(17,0,2));games[0].apple=(0,0)
        for _ in range(3):stagnation=pool.step([(0,1)])[0][0]
        assert stagnation.truncated and stagnation.reason=='stagnation' and stagnation.reward==config.stagnation_reward
        pool.reset(0,episode_seed(17,0,3));assert not pool.episodes[0].done
        assert all(g.get_state()==state for g,state in zip(games[1:],initial[1:]))
    finally:pool.close()


@pytest.mark.parametrize('count',(1,2,4,8))
@pytest.mark.parametrize('encoder',(LEGACY,DISTANCE,BINARY))
@pytest.mark.parametrize('strategy',('vectorized','multiprocessing'))
def test_audit_pool_matches_independent_oracles_with_asynchronous_resets(count,encoder,strategy):
    config=TrainingConfig(episodes=100,stagnation_factor=2)
    pool=EnvironmentPool(config,encoder,count,strategy)
    oracles=[Episode(config,encoder=None if encoder==LEGACY else get_encoder(encoder)) for _ in range(count)]
    sequences=[0]*count;completed=0;transitions=0;lengths=[];first_completion={}
    try:
        for i in range(count):
            seed=episode_seed(19,i,0)
            assert pool.reset(i,seed)==oracles[i].reset(seed)
        for tick in range(25):
            indices=[i for i in range(count) if i==0 or tick%2==0]
            actions=[(i,0 if i==0 else 1) for i in indices]
            results=pool.step(actions)
            for (i,action),(transition,score,apples) in zip(actions,results):
                expected=oracles[i].step(action)
                assert transition==expected and score==oracles[i].game.score and apples==oracles[i].game.apples_eaten
                transitions+=1
                if transition.done:
                    first_completion.setdefault(i,tick)
                    lengths.append(oracles[i].steps);completed+=1;sequences[i]+=1
                    seed=episode_seed(19,i,sequences[i])
                    assert pool.reset(i,seed)==oracles[i].reset(seed)
        assert transitions==sum(1 for tick in range(25) for i in range(count) if i==0 or tick%2==0)
        assert completed>0
        if count>1:assert first_completion[0]!=first_completion[1]
        if strategy=='vectorized':
            assert all(a.game.rng.getstate()==b.game.rng.getstate() for a,b in zip(pool.episodes,oracles))
    finally:pool.close()


@pytest.mark.parametrize('count',(1,2,4,8))
@pytest.mark.parametrize('policy',('transition','vector'))
def test_audit_nondivisible_global_target_schedule_and_replay_mapping(count,policy):
    trainer=ParallelTrainer(TrainingConfig(episodes=100,batch_size=2,target_sync=3),FrontierConfig(environments=count,update_policy=policy))
    sync=[];original=trainer.target.load_state_dict
    def load(state,*args,**kwargs):sync.append(trainer.total_steps);return original(state,*args,**kwargs)
    trainer.target.load_state_dict=load
    try:
        for _ in range(4):
            before=trainer.total_steps;batch=trainer.train_batch()
            assert trainer.replay.items[before:trainer.total_steps]==batch
        assert sync==list(range(3,trainer.total_steps+1,3))
        assert trainer.total_steps==4*count
        assert trainer.optimizer_updates==(max(0,4*count-1) if policy=='transition' else 4-(count==1))
    finally:trainer.close()


def wait_finished(worker):
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        snapshot=worker.latest()
        if snapshot and snapshot.status in ('Failed','Completed','Save failed'):return snapshot
        time.sleep(.005)
    pytest.fail('audit worker timeout')


def test_audit_recovery_claim_requires_a_resumable_checkpoint(tmp_path,monkeypatch):
    original=ParallelTrainer.optimize;calls=[]
    def fail_second(self):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('audit optimizer failure after a completed transition')
        return original(self)
    monkeypatch.setattr(ParallelTrainer,'optimize',fail_second)
    worker=FrontierWorker(tmp_path/'profile',tmp_path/'experiments');restored=None
    try:
        worker.start(TrainingConfig(episodes=10,episode_cap=1,batch_size=2),FrontierConfig(environments=4))
        failed=wait_finished(worker)
        assert failed.status=='Failed' and 'unsafe' in failed.message
        assert failed.checkpoint is None and 'no verified checkpoint' in failed.message
        assert not CheckpointStore(tmp_path/'profile/checkpoints').entries()
    finally:
        worker.shutdown()
        if restored:restored.close()


def test_audit_completed_counters_match_history_after_mid_batch_failure(tmp_path,monkeypatch):
    original=ParallelTrainer.optimize;calls=[]
    def fail_second(self):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('audit interrupted batch')
        return original(self)
    monkeypatch.setattr(ParallelTrainer,'optimize',fail_second)
    worker=FrontierWorker(tmp_path/'profile',tmp_path/'experiments')
    try:
        worker.start(TrainingConfig(episodes=10,episode_cap=1,batch_size=2),FrontierConfig(environments=4))
        failed=wait_finished(worker);worker.shutdown()
        root=Path(failed.experiment)
        assert failed.metrics['completed']==1
        assert len(HistoryStore(root/'history').load(root.name).records)==failed.metrics['completed']
    finally:worker.shutdown()
