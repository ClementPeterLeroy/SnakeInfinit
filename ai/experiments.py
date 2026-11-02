"""Versioned experiment descriptions; execution remains the existing 11-input DQN."""
from dataclasses import asdict, dataclass
from datetime import datetime,timezone
import json
from pathlib import Path
from uuid import uuid4

BASELINE_ENCODER='snake-classic-binary11-v1'


@dataclass(frozen=True)
class Experiment:
    identifier: str
    name: str
    encoder_id: str
    input_count: int
    architecture: tuple[int,...]
    hyperparameters: dict
    reward_configuration: dict
    training_seed: int
    evaluation_seeds: tuple[int,...]
    environment_step_budget: int
    training_runs: tuple[str,...] = ()
    checkpoints: tuple[str,...] = ()
    evaluation_results: tuple[str,...] = ()

    def __post_init__(self):
        if not self.name or not self.encoder_id or type(self.input_count) is not int or self.input_count<1:
            raise ValueError('Experiment requires a name and explicit encoder/input dimensions')
        if len(self.architecture)<2 or self.architecture[0]!=self.input_count or any(type(n) is not int or n<1 for n in self.architecture):
            raise ValueError('Experiment architecture must match its declared input dimension')
        if type(self.environment_step_budget) is not int or self.environment_step_budget<1:
            raise ValueError('Environment-step budget must be positive')
        if any(type(s) is not int or not 0<=s<2**63 for s in self.evaluation_seeds):
            raise ValueError('Invalid evaluation seed set')


class ExperimentStore:
    def __init__(self,root):
        self.root=Path(root)

    def save(self,experiment):
        if not experiment.identifier.isalnum() or len(experiment.identifier)>64:
            raise ValueError('Invalid experiment identifier')
        self.root.mkdir(parents=True,exist_ok=True)
        path=self.root/(experiment.identifier+'.json')
        with path.open('x',encoding='utf-8') as stream:
            json.dump(dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),**asdict(experiment)),stream,indent=2,allow_nan=False)
        return path

    def baseline(self,name,network,config,run,checkpoint):
        experiment=Experiment(uuid4().hex,name,BASELINE_ENCODER,11,network.config.sizes,
            config.to_dict(),{key:getattr(config,key) for key in ('food_reward','collision_reward','step_reward','stagnation_reward')},
            config.seed,tuple(range(25)),config.episodes*config.episode_cap,(run,),(str(checkpoint),))
        self.save(experiment)
        return experiment.identifier

    def link(self,identifier,kind,value):
        if not identifier.isalnum() or kind not in ('checkpoint','evaluation'):
            raise ValueError('Invalid experiment linkage')
        with (self.root/(identifier+'.events.jsonl')).open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(kind=kind,value=str(value),created_utc=datetime.now(timezone.utc).isoformat()))+'\n')
            stream.flush()

    def load(self,identifier):
        if not identifier.isalnum():
            raise ValueError('Invalid experiment identifier')
        data=json.loads((self.root/(identifier+'.json')).read_text(encoding='utf-8'))
        data['links']=[]
        events=self.root/(identifier+'.events.jsonl')
        if events.is_file():
            for line in events.read_text(encoding='utf-8').splitlines():
                try:data['links'].append(json.loads(line))
                except ValueError:continue  # An interrupted final append never invents a link.
        return data
