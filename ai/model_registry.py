"""Small metadata-first architecture registry; Torch constructors are loaded on demand."""
from dataclasses import asdict,dataclass
from ai.encoders import get_encoder,LEGACY,DISTANCE,BINARY,ORDERED


@dataclass(frozen=True)
class Architecture:
    identifier: str
    name: str
    family: str
    encoders: tuple
    schema: int = 1
    actions: tuple = ('Straight','Turn left','Turn right')

    @property
    def observation_shapes(self):return {encoder:get_encoder(encoder).shape for encoder in self.encoders}

    def parameter_count(self,spec):
        if spec.architecture_id!=self.identifier:raise ValueError('Architecture configuration mismatch')
        return spec.parameter_count

    def construct(self,spec):
        if spec.architecture_id!=self.identifier:raise ValueError('Architecture configuration mismatch')
        if spec.configuration_id=='A':return build_model(spec)
        from ai.registered_networks import RegisteredNetwork
        return RegisteredNetwork(spec)


ARCHITECTURES={
    'mlp_11_16_8_3':Architecture('mlp_11_16_8_3','Legacy MLP 11 → 16 → 8 → 3','MLP',(LEGACY,)),
    'mlp_distance19':Architecture('mlp_distance19','Distance-aware MLP','MLP',(DISTANCE,)),
    'cnn_spatial_compact':Architecture('cnn_spatial_compact','Compact spatial CNN','CNN',(BINARY,ORDERED)),
}
LINEUP={
    'A':('Legacy DQN — 11 Inputs','mlp_11_16_8_3',LEGACY,(16,8),'snake_dqn_11'),
    'B':('Distance DQN — 19 Inputs','mlp_distance19',DISTANCE,(64,32),'snake_dqn_distance19'),
    'C':('Spatial CNN — Binary Body','cnn_spatial_compact',BINARY,(),'snake_cnn_binary'),
    'D':('Spatial CNN — Ordered Body','cnn_spatial_compact',ORDERED,(),'snake_cnn_ordered'),
}


def get_architecture(identifier):
    try:return ARCHITECTURES[identifier]
    except (KeyError,TypeError):raise ValueError('Unsupported architecture: '+str(identifier)) from None


@dataclass(frozen=True)
class ModelSpec:
    configuration_id: str = 'B'
    hidden_sizes: tuple = (64,32)
    seed: int | None = 0
    architecture_id: str = 'mlp_distance19'
    encoder_id: str = DISTANCE
    schema_version: int = 1

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version!=1 or self.configuration_id not in LINEUP:raise ValueError('Unsupported model schema/configuration')
        architecture=get_architecture(self.architecture_id)
        encoder=get_encoder(self.encoder_id)
        if self.encoder_id not in architecture.encoders or self.architecture_id not in encoder.architectures:
            raise ValueError('Incompatible architecture and observation encoder')
        expected=LINEUP[self.configuration_id]
        if (self.architecture_id,self.encoder_id)!=(expected[1],expected[2]):raise ValueError('Model configuration identity mismatch')
        if self.seed is not None and (type(self.seed) is not int or not 0<=self.seed<2**63):raise ValueError('Invalid initialization seed')
        if not isinstance(self.hidden_sizes,tuple):raise ValueError('Hidden sizes must be a tuple')
        if self.configuration_id=='B' and (len(self.hidden_sizes) not in (2,3) or any(type(n) is not int or not 2<=n<=512 for n in self.hidden_sizes)):
            raise ValueError('Distance MLP needs 2–3 hidden layers with 2..512 neurons')
        if self.configuration_id in ('C','D') and self.hidden_sizes:raise ValueError('The shared CNN architecture has fixed layers')
        if self.configuration_id=='A' and self.hidden_sizes!=(16,8):raise ValueError('Configuration A uses the original 16,8 hidden layers')

    @property
    def input_size(self):return 11 if self.configuration_id=='A' else 19 if self.configuration_id=='B' else 1879
    @property
    def output_size(self):return 3
    @property
    def activation(self):return 'relu'
    @property
    def sizes(self):return (self.input_size,*self.hidden_sizes,3) if self.configuration_id in ('A','B') else (1879,32,64,64,128,3)
    @property
    def parameter_count(self):return sum((a+1)*b for a,b in zip(self.sizes,self.sizes[1:])) if self.configuration_id in ('A','B') else 188419
    def to_dict(self):return asdict(self)

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict) or set(data)!=set(cls.__dataclass_fields__):raise ValueError('Unsupported model specification fields')
        if not isinstance(data['hidden_sizes'],list):raise ValueError('Serialized hidden sizes must be a list')
        return cls(**{**data,'hidden_sizes':tuple(data['hidden_sizes'])})


def model_spec(configuration='A',seed=0,hidden=None):
    if configuration not in LINEUP:raise ValueError('Unsupported model configuration')
    _,architecture,encoder,defaults,_=LINEUP[configuration]
    return ModelSpec(configuration,defaults if hidden is None else tuple(hidden),seed,architecture,encoder)


def build_model(spec):
    if spec.configuration_id=='A':
        from ai.config import NetworkConfig
        from ai.network import SnakeNetwork
        return SnakeNetwork(NetworkConfig(seed=spec.seed))
    return get_architecture(spec.architecture_id).construct(spec)


def identity(config):
    from ai.config import NetworkConfig
    if isinstance(config,ModelSpec) and config.configuration_id=='A':config=NetworkConfig(seed=config.seed)
    legacy=isinstance(config,NetworkConfig)
    spec=model_spec('A',config.seed) if legacy else config
    encoder=get_encoder(spec.encoder_id)
    parameters=config.to_dict() if legacy else (dict(channels=[3,32,64,64],kernel_size=3,padding=1,strides=[2,2,2],
        flatten=1024,heading=4,dense_hidden=128,activation='relu',outputs=3) if spec.configuration_id in ('C','D') else
        dict(input_features=19,hidden_sizes=list(spec.hidden_sizes),activation='relu',outputs=3))
    shape={k:list(v) for k,v in encoder.shape.items()} if isinstance(encoder.shape,dict) else list(encoder.shape)
    return dict(model_configuration_id=spec.configuration_id,architecture_id=spec.architecture_id,encoder_id=spec.encoder_id,
        architecture_name=('Legacy MLP '+' → '.join(map(str,config.sizes))) if legacy else get_architecture(spec.architecture_id).name,encoder_name=encoder.name,
        network_family=get_architecture(spec.architecture_id).family,architecture_schema_version=1,encoder_schema_version=1,architecture_parameters=parameters,
        actions=list(get_architecture(spec.architecture_id).actions),
        encoder_configuration=dict(board_width=25,board_height=25,normalization=encoder.normalization),input_shape=shape,
        parameter_count=sum((a+1)*b for a,b in zip(config.sizes,config.sizes[1:])) if legacy else spec.parameter_count)


def parse_model_config(metadata):
    if metadata.get('format_version')==1:
        from ai.config import NetworkConfig
        config=NetworkConfig.from_dict(metadata.get('architecture'))
        if metadata.get('encoder_id',LEGACY)!=LEGACY or metadata.get('architecture_id','mlp_11_16_8_3')!='mlp_11_16_8_3':
            raise ValueError('Incompatible legacy model identity')
        if metadata.get('encoder_schema_version',1)!=1 or metadata.get('architecture_schema_version',1)!=1:raise ValueError('Unsupported legacy identity schema')
        return config
    if metadata.get('format_version')==2:
        spec=ModelSpec.from_dict(metadata.get('architecture'))
        if spec.configuration_id=='A':raise ValueError('Legacy models use their original format')
        expected=identity(spec)
        for key,value in expected.items():
            # Checkpoint envelopes may defer action ordering to their validated network.json.
            if key=='actions' and key not in metadata and metadata.get('kind') in ('resumable-boundary','model-snapshot'):continue
            if metadata.get(key)!=value:raise ValueError('Incompatible model metadata: '+key)
        return spec
    raise ValueError('Unsupported model format')
