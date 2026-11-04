"""One DQN learner, deterministic ordered collectors and batched epsilon-greedy policy."""
from dataclasses import dataclass
import math
import random
import torch
from ai.dqn import Trainer
from ai.environment_pool import EnvironmentPool,episode_seed
from ai.encoders import LEGACY
from ai.frontier_config import FrontierConfig


@dataclass(frozen=True)
class Completion:
    index: int
    environment: int
    sequence: int
    seed: int
    score: int
    length: int
    reward: float
    reason: str
    steps: int
    epsilon: float
    updates: int
    mean_loss: float | None
    loss: float | None
    gradient_norm: float | None


class ParallelTrainer(Trainer):
    def __init__(self,config,execution=FrontierConfig(environments=4),network=None,architecture=None,learner=None):
        if learner is None:super().__init__(config,network,architecture)
        else:self.__dict__.update(learner.__dict__)
        self.execution=execution
        self.pool=EnvironmentPool(config,self.online.config.encoder_id if self.codec else LEGACY,
            execution.environments,execution.resolved_strategy)
        self.slots=[dict(sequence=-1,seed=None,observation=None,actions=[],score=0,reward=0.,length=0,
            updates=0,loss_sum=0.,rng=random.Random(episode_seed(config.seed,i,2**32))) for i in range(execution.environments)]
        self.completions=[]
        self.vector_steps=0
        self.at_boundary=True  # Quiescent vector boundary: all partial episodes are serializable.

    @property
    def active_environments(self):return sum(slot['observation'] is not None for slot in self.slots)

    def _start(self,index):
        slot=self.slots[index]
        slot.update(sequence=slot['sequence']+1,actions=[],score=0,reward=0.,length=0,updates=0,loss_sum=0.)
        slot['seed']=episode_seed(self.config.seed,index,slot['sequence'])
        slot['observation']=self.pool.reset(index,slot['seed'])

    def choose_actions(self,indices):
        actions={};greedy=[]
        for offset,index in enumerate(indices):
            # Each transition has a globally indexed epsilon, independent of vector width.
            progress=min(1.,(self.total_steps+offset)/self.config.epsilon_decay_steps)
            epsilon=self.config.epsilon_start+progress*(self.config.epsilon_min-self.config.epsilon_start)
            rng=self.slots[index]['rng']
            if rng.random()<epsilon:actions[index]=rng.randrange(3)
            else:greedy.append(index)
        if greedy:
            observations=[self.slots[index]['observation'] for index in greedy]
            inputs=torch.tensor(observations,dtype=torch.float32) if self.codec is None else self.codec.stack(observations)
            with torch.no_grad():values=self.online(inputs)
            if values.shape!=(len(greedy),3) or not torch.isfinite(values).all():raise ValueError('Invalid batched policy Q-values')
            actions.update(zip(greedy,values.argmax(1).tolist()))
        return [(index,actions[index]) for index in indices]

    def train_batch(self,max_transitions=None,optimize=True,completion_limit=None):
        self.completions=[]
        remaining=self.config.episodes-self.completed
        if remaining<=0:return []
        # Reserve at most the remaining episode slots: target cannot overshoot.
        active=[i for i,s in enumerate(self.slots) if s['observation'] is not None]
        for index,slot in enumerate(self.slots):
            if len(active)>=remaining:break
            if slot['observation'] is None:self._start(index);active.append(index)
        # If a resumed target was shortened, retain surplus partial episodes
        # without advancing/counting them. They can continue on a later extension.
        indices=sorted(active)[:remaining]
        if completion_limit is not None:indices=indices[:max(0,completion_limit-self.completed)]
        if max_transitions is not None:indices=indices[:max_transitions]
        if not indices:return []
        actions=self.choose_actions(indices)
        results=self.pool.step(actions)
        transitions=[result[0] for result in results]
        self.vector_steps+=1
        for offset,((index,action),transition) in enumerate(zip(actions,transitions)):
            slot=self.slots[index]
            slot['actions'].append(action);slot['observation']=transition.next_observation
            slot['length']+=1;slot['reward']+=transition.reward
            score=results[offset][1]
            ate=score-slot['score'];slot['score']=score
            self.replay.append(transition);self.total_steps+=1;self.apples+=ate
            self.cumulative_reward+=transition.reward
            if optimize and (self.execution.update_policy=='transition' or offset==len(transitions)-1):
                previous=self.optimizer_updates;self.optimize()
                if self.optimizer_updates>previous:
                    slot['updates']+=1;slot['loss_sum']+=self.loss
            if self.total_steps%self.config.target_sync==0:self.target.load_state_dict(self.online.state_dict())
            self.epsilons.append(self.epsilon)
            if transition.done:
                self.completed+=1;self.best=max(self.best,slot['score'])
                self.scores.append(slot['score']);self.lengths.append(slot['length']);self.rewards.append(slot['reward'])
                self.last_reason=transition.reason
                self.collisions+=int(transition.terminated and transition.reason!='board_completion')
                self.stagnations+=int(transition.reason=='stagnation')
                self.completions.append(Completion(self.completed,index,slot['sequence'],slot['seed'],slot['score'],
                    slot['length'],slot['reward'],transition.reason,self.total_steps,self.epsilon,slot['updates'],
                    slot['loss_sum']/slot['updates'] if slot['updates'] else None,self.loss,self.gradient_norm))
                slot['observation']=None
        return transitions

    def train_step(self):return self.train_batch()

    def metrics(self):
        metrics=super().metrics()
        metrics.update(score=max((s['score'] for s in self.slots if s['observation'] is not None),default=0),
            environments=self.execution.environments,active_environments=self.active_environments,
            strategy=self.execution.resolved_strategy,update_policy=self.execution.update_policy,
            optimizer_updates=self.optimizer_updates,vector_steps=self.vector_steps,worker_health='Healthy',
            updates_per_transition=self.optimizer_updates/max(1,self.total_steps))
        return metrics

    def snapshot_state(self):
        return dict(schema=1,execution=self.execution.to_dict(),vector_steps=self.vector_steps,
            slots=[{**{k:v for k,v in slot.items() if k not in ('rng','observation')},
                'active':slot['observation'] is not None,'rng':slot['rng'].getstate()} for slot in self.slots])

    def restore_state(self,state):
        if not isinstance(state,dict) or state.get('schema')!=1 or len(state['slots'])!=len(self.slots):raise ValueError('Invalid parallel state')
        if type(state['vector_steps']) is not int or not 0<=state['vector_steps']<=self.total_steps:raise ValueError('Invalid vector counter')
        self.vector_steps=state['vector_steps']
        for index,saved in enumerate(state['slots']):
            sequence=saved['sequence'];actions=saved['actions']
            if type(sequence) is not int or sequence< -1 or not isinstance(actions,list) or len(actions)>self.config.episode_cap or any(type(a) is not int or a not in (0,1,2) for a in actions):raise ValueError('Invalid parallel episode trace')
            if type(saved['active']) is not bool:raise ValueError('Invalid parallel active flag')
            if sequence>=0 and saved['seed']!=episode_seed(self.config.seed,index,sequence):raise ValueError('Invalid parallel episode seed')
            if any(type(saved[k]) is not int or saved[k]<0 for k in ('score','length','updates')) or any(not math.isfinite(saved[k]) for k in ('reward','loss_sum')):raise ValueError('Invalid parallel counters')
            slot=self.slots[index]
            slot.update({k:v for k,v in saved.items() if k not in ('rng','active')})
            slot['rng'].setstate(saved['rng']);slot['observation']=None
            if saved['active']:
                observation=self.pool.reset(index,saved['seed'])
                reward=score=0
                for action in actions:
                    transition,score,_=self.pool.step([(index,action)])[0]
                    if transition.done:raise ValueError('Active episode trace is terminal')
                    observation=transition.next_observation;reward+=transition.reward
                if saved['length']!=len(actions) or saved['score']!=score or saved['reward']!=reward:raise ValueError('Parallel episode trace counters mismatch')
                slot['observation']=observation

    def close(self,measure=False):
        if measure and not self.pool.closed:
            try:self.pool.usage()
            except (RuntimeError,OSError,EOFError):pass
        self.pool.close()

    def restart_collectors(self):
        if not self.pool.closed:return
        state=self.snapshot_state()
        self.pool=EnvironmentPool(self.config,self.online.config.encoder_id if self.codec else LEGACY,
            self.execution.environments,self.execution.resolved_strategy)
        try:self.restore_state(state)
        except BaseException:self.pool.close();raise
        if hasattr(self,'_profile_pool'):self._profile_pool()
