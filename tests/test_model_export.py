"""Project model snapshots preserve sources and never overwrite existing exports."""
import hashlib
import json
import pytest
from tools.export_models import export_latest,export_model


def fixture_model(folder,episode):
    folder.mkdir(parents=True)
    filename='weights-'+'a'*32+'.pt'
    weights=f'weights at episode {episode}'.encode()
    (folder/filename).write_bytes(weights)
    (folder/'network.json').write_text(json.dumps(dict(format_version=1,weights_file=filename,
        sha256=hashlib.sha256(weights).hexdigest())),encoding='utf-8')
    metadata=dict(format_version=1,kind='resumable-boundary',model_identifier='run123',
        model_name='Snake DQN Alpha',episode=episode,environment_steps=episode*10,
        architecture={'input_size':11},hyperparameters={'seed':0},created_utc='2026-11-02T00:00:00+00:00')
    (folder/'training.json').write_text(json.dumps(metadata),encoding='utf-8')
    (folder/'training.pt').write_bytes(b'private resume state')


def test_latest_model_export_is_visible_portable_and_read_only(tmp_path):
    profile=tmp_path/'profile'
    fixture_model(profile/'checkpoints/run123/early',10)
    fixture_model(profile/'checkpoints/run123/latest',20)
    before={str(p):p.read_bytes() for p in profile.rglob('*') if p.is_file()}
    output=tmp_path/'project/models'
    exported=export_latest(profile,output)
    assert len(exported)==1 and exported[0].name=='Snake_DQN_Alpha-ep20-run123'
    assert len(list(exported[0].iterdir()))==3
    assert not (exported[0]/'training.pt').exists()
    card=json.loads((exported[0]/'model_card.json').read_text())
    assert card['episode']==20 and card['resumable'] is False
    assert before=={str(p):p.read_bytes() for p in profile.rglob('*') if p.is_file()}
    assert export_latest(profile,output)==exported
    (exported[0]/'network.json').write_text('existing change')
    with pytest.raises(FileExistsError):export_latest(profile,output)
    assert (exported[0]/'network.json').read_text()=='existing change'


def test_export_rejects_corrupt_weights_before_creating_folder(tmp_path):
    source=tmp_path/'source'
    fixture_model(source,1)
    next(source.glob('weights-*.pt')).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):export_model(source,tmp_path/'output')
    assert not (tmp_path/'output').exists()
