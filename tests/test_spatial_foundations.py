"""Encoder geometry, registry contracts, structured replay and model persistence."""
from dataclasses import replace
import json
import math
import random
import pytest
from core.game import Game
from core.types import Direction
from ai.encoders import get_encoder,LEGACY,DISTANCE,BINARY,ORDERED
from ai.model_registry import model_spec,build_model,get_architecture,identity,ModelSpec
from ai.observations import encode_observation


def state(body=((12,12),(11,12),(10,12)),food=(20,12),direction=Direction.RIGHT):
    return replace(Game(seed=0).get_state(),snake=body,apple=food,direction=direction,pending_direction=None)


def test_distance_compass_walls_body_diagonal_and_food():
    initial=state(((0,0),(0,1),(0,2)),(2,0))
    result=get_encoder(DISTANCE).encode(initial)
    assert result[:11]==encode_observation(initial) and len(result)==19
    assert result[11:]==(0.,0.,1.,1.,0.,0.,0.,0.)
    obstacles=state(((12,12),(15,12),(15,9)))
    values=get_encoder(DISTANCE).encode(obstacles)
    assert values[12]==pytest.approx(2/24) and values[13]==pytest.approx(2/24)
    adjacent=state(((12,12),(13,12)))
    assert get_encoder(DISTANCE).encode(adjacent)[13]==0
    assert get_encoder(DISTANCE).encode(obstacles)==values
    assert all(0<=value<=1 for value in values)


@pytest.mark.parametrize('encoder',(BINARY,ORDERED))
def test_spatial_channels_heading_and_empty_cells(encoder):
    observation=get_encoder(encoder).encode(state(direction=Direction.UP))
    board=observation.spatial
    assert len(board)==3 and len(board[0])==25 and len(board[0][0])==25
    assert board[1][12][12]==1 and board[0][12][12]==0
    assert board[2][12][20]==1 and observation.heading==(1.,0.,0.,0.)
    assert all(channel[0][0]==0 for channel in board)
    assert board[0][12][11]==1
    assert board[0][12][10]==(1 if encoder==BINARY else .5)


@pytest.mark.parametrize('n',(0,1,2,50,624))
def test_ordered_body_short_and_long_snakes(n):
    positions=tuple((index%25,index//25) for index in range(n+1))
    observation=get_encoder(ORDERED).encode(state(positions,None))
    board=observation.spatial
    values=[board[0][y][x] for x,y in positions[1:]]
    assert all(math.isfinite(value) and 0<value<=1 for value in values)
    assert len(set(values))==n
    if n:assert values[0]==1 and values[-1]==pytest.approx(1/n)
    assert sum(sum(row) for row in board[1])==1
    assert sum(sum(row) for row in board[2])==0


def test_registry_compatibility_unknown_ids_and_serialization():
    with pytest.raises(ValueError):get_encoder('unknown')
    with pytest.raises(ValueError):get_architecture('unknown')
    with pytest.raises(ValueError):ModelSpec('B',(64,32),0,'mlp_distance19',BINARY)
    with pytest.raises(ValueError):model_spec('B',hidden=(32,))
    with pytest.raises(ValueError):get_architecture('cnn_spatial_compact').construct(model_spec('A'))
    for configuration in 'ABCD':
        spec=model_spec(configuration,7)
        assert ModelSpec.from_dict(json.loads(json.dumps(spec.to_dict())))==spec
        assert identity(spec)['encoder_id']==spec.encoder_id


def test_all_networks_forward_backward_parameter_counts_and_shared_cnn():
    torch=pytest.importorskip('torch')
    torch.set_num_threads(1)
    from ai.registered_networks import ObservationCodec
    models={key:build_model(model_spec(key,7)) for key in 'ABCD'}
    assert {key:model.parameter_count for key,model in models.items()}==dict(A=355,B=3459,C=188419,D=188419)
    assert all(torch.equal(a,b) for a,b in zip(models['C'].parameters(),models['D'].parameters()))
    for key,model in models.items():
        spec=model_spec(key,7)
        observation=get_encoder(spec.encoder_id).encode(state())
        inputs=torch.tensor([observation]*2) if key=='A' else ObservationCodec(spec).stack((observation,observation))
        outputs=model(inputs)
        assert outputs.shape==(2,3) and torch.isfinite(outputs).all()
        outputs.square().mean().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    wider=build_model(model_spec('B',hidden=(32,16,8)))
    assert wider.config.sizes==(19,32,16,8,3)


@pytest.mark.parametrize('configuration','ABCD')
def test_training_checkpoint_resume_model_export_and_arena(configuration,tmp_path):
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    from ai.dqn import Trainer
    from ai.training_config import TrainingConfig
    from ai.checkpoints import CheckpointStore
    from ai.persistence import ModelStore
    from ai.arena import Arena
    trainer=Trainer(TrainingConfig(episodes=6,episode_cap=3,batch_size=2,target_sync=3),architecture=model_spec(configuration,4))
    before=[p.clone() for p in trainer.online.parameters()]
    for _ in range(6):trainer.train_step()
    assert trainer.completed==2 and math.isfinite(trainer.loss)
    assert any(not torch.equal(a,b) for a,b in zip(before,trainer.online.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(trainer.online.parameters(),trainer.target.parameters()))
    store=CheckpointStore(tmp_path/'checkpoints')
    path=store.save('run','Research model',trainer)
    loaded,metadata=store.load_model(path)
    assert metadata['model_configuration_id']==configuration
    resumed,_=store.resume(path)
    for _ in range(6):assert trainer.train_step()==resumed.train_step()
    assert all(torch.equal(a,b) for a,b in zip(trainer.online.parameters(),resumed.online.parameters()))
    arena=Arena(loaded,TrainingConfig(episode_cap=3),seed=42)
    first=[]
    while not arena.environment.done:first.append(arena.step())
    arena.reset(42)
    second=[]
    while not arena.environment.done:second.append(arena.step())
    assert first==second and all(p.grad is None for p in arena.network.parameters())


def test_replay_rejects_encoder_mismatch_and_invalid_spatial_cells():
    pytest.importorskip('torch')
    from ai.registered_networks import ObservationCodec
    codec=ObservationCodec(model_spec('C'))
    packed=codec.pack((get_encoder(BINARY).encode(state()),))
    with pytest.raises(ValueError):ObservationCodec(model_spec('D')).unpack(packed,1)
    packed['positions']=[b'\xff\xff']
    with pytest.raises(ValueError):codec.unpack(packed,1)


def test_old_model_without_registry_fields_has_identical_actions(tmp_path):
    torch=pytest.importorskip('torch')
    from ai.persistence import ModelStore
    original=build_model(model_spec('A',17))
    store=ModelStore(tmp_path)
    folder=store.save('historical',original)
    path=folder/'network.json'
    metadata=json.loads(path.read_text())
    # An actual format-1 envelope does not contain the new registry identity.
    for key in identity(original.config):
        if key!='actions':metadata.pop(key,None)
    path.write_text(json.dumps(metadata))
    loaded,_=store.load('historical')
    for seed in range(10):
        observation=torch.tensor(encode_observation(Game(seed=seed).get_state()))
        assert torch.equal(original(observation),loaded(observation))
        assert original(observation).argmax()==loaded(observation).argmax()


@pytest.mark.parametrize('field,value',(('architecture_id','unknown'),('encoder_id','unknown'),
    ('encoder_schema_version',99),('architecture_schema_version',99),('parameter_count',1),
    ('actions',['Turn left','Straight','Turn right'])))
def test_model_metadata_rejects_unknown_or_incompatible_identity(field,value,tmp_path):
    pytest.importorskip('torch')
    from ai.persistence import ModelStore
    store=ModelStore(tmp_path)
    folder=store.save('spatial',build_model(model_spec('C')))
    path=folder/'network.json'
    metadata=json.loads(path.read_text());metadata[field]=value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):store.load('spatial')


def test_runtime_resolves_registries_once_and_passes_spatial_heading(monkeypatch):
    torch=pytest.importorskip('torch')
    from ai.dqn import Trainer
    from ai.arena import Arena
    from ai.training_config import TrainingConfig
    trainer=Trainer(TrainingConfig(batch_size=2,episode_cap=4),architecture=model_spec('D'))
    arena=Arena(build_model(model_spec('D')),seed=12)
    expected=arena.codec.single(arena.environment.observation())
    inputs=[]
    hook=arena.network.register_forward_pre_hook(lambda module,args:inputs.append(args[0]))
    def forbidden(*args):raise AssertionError('Registry lookup in the simulation loop')
    for module in ('ai.encoders','ai.model_registry'):
        monkeypatch.setattr(module+'.get_encoder',forbidden)
    monkeypatch.setattr('ai.model_registry.get_architecture',forbidden)
    for _ in range(5):trainer.train_step()
    arena.step();hook.remove()
    assert len(inputs)==1
    assert inputs[0]['spatial'].shape==expected['spatial'].shape
    assert torch.equal(inputs[0]['spatial'],expected['spatial'])
    assert torch.equal(inputs[0]['heading'],expected['heading'])
