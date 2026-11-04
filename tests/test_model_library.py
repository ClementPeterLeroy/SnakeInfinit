from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pytest
from ai.model_library import ModelLibrary,FAMILIES


def checkpoint(profile,run='original',episode=1,family='A'):
    torch=pytest.importorskip('torch')
    from ai.dqn import Trainer
    from ai.training_config import TrainingConfig
    from ai.model_registry import model_spec
    from ai.checkpoints import CheckpointStore
    trainer=Trainer(TrainingConfig(episodes=10,episode_cap=1,batch_size=2),architecture=model_spec(family))
    for _ in range(episode):trainer.train_step()
    return CheckpointStore(profile/'checkpoints').save(run,'Old name',trainer)


def test_unique_independent_generation_numbers_concurrent_and_persistent(tmp_path):
    library=ModelLibrary(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        identifiers=list(pool.map(lambda i:ModelLibrary(tmp_path).create(tuple(FAMILIES)[i%4]),range(20)))
    assert len(set(identifiers))==20
    restarted=ModelLibrary(tmp_path)
    for family in FAMILIES:
        assert [g['number'] for g in restarted.generations(family)]==list(range(1,6))
        assert all(g['name'].startswith('Snake DQN '+family+' — Gen ') for g in restarted.generations(family))
    generation=identifiers[0];old=restarted.get(generation)['name']
    restarted.describe(generation,'Lower learning rate')
    assert restarted.get(generation)['name']==old


def test_legacy_migration_is_metadata_only_idempotent_and_groups_real_run(tmp_path,monkeypatch):
    first=checkpoint(tmp_path,episode=1);second=checkpoint(tmp_path,episode=2)
    before={str(p):p.read_bytes() for folder in (first,second) for p in folder.iterdir()}
    library=ModelLibrary(tmp_path)
    import torch
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Listing loaded tensors'))
    summary=library.scan()
    assert summary['generations_added']==1 and summary['checkpoints_added']==2
    assert len(library.generations('Alpha'))==1
    assert library.scan()['generations_added']==0
    assert before=={str(p):p.read_bytes() for folder in (first,second) for p in folder.iterdir()}


def test_resume_keeps_generation_and_interface_is_only_metadata(tmp_path):
    path=checkpoint(tmp_path)
    library=ModelLibrary(tmp_path);library.scan()
    cp=library.find_checkpoint(path);generation=cp['generation_id']
    assert library.register_run('newsession','Alpha',resume=path,interface='Frontier')==generation
    assert library.register_run('legacyagain','Alpha',generation=generation,interface='Legacy')==generation
    assert len(library.generations())==1
    with pytest.raises(ValueError):library.register_run('bad','Distance',generation=generation)


def test_delete_one_then_generation_restore_and_never_reuse_number(tmp_path):
    first=checkpoint(tmp_path,episode=1);second=checkpoint(tmp_path,episode=2)
    library=ModelLibrary(tmp_path);library.scan()
    cp=library.find_checkpoint(first);gid=cp['generation_id']
    preview=library.preview_delete('checkpoint',cp['id'])
    op=library.delete(preview)
    assert not first.exists() and second.exists()
    with pytest.raises(ValueError):library.resolve(cp['id'])
    with pytest.raises(ValueError):library.delete(preview)
    library.restore(op);assert first.exists() and library.resolve(cp['id'])
    op=library.delete(library.preview_delete('generation',gid))
    assert not first.exists() and not second.exists() and not library.generations()
    assert library.get(library.create('Alpha'))['number']==2
    library.restore(op)
    assert first.exists() and second.exists() and library.get(gid)['number']==1


def test_loaded_and_training_deletion_guards(tmp_path):
    path=checkpoint(tmp_path);library=ModelLibrary(tmp_path);library.scan()
    cp=library.find_checkpoint(path)
    for training in (False,True):
        lease=library.acquire(cp['generation_id'],'Test viewer',cp['id'],training=training)
        with pytest.raises(ValueError,match='active or loaded'):library.preview_delete('checkpoint',cp['id'])
        with pytest.raises(ValueError):library.preview_delete('generation',cp['generation_id'])
        library.release(lease)
    assert library.preview_delete('checkpoint',cp['id'])


def test_missing_corrupt_incompatible_and_stale_entries_are_not_substituted(tmp_path):
    path=checkpoint(tmp_path);library=ModelLibrary(tmp_path);library.scan()
    cp=library.find_checkpoint(path);data=json.loads((path/'network.json').read_text())
    weights=path/data['weights_file'];payload=weights.read_bytes();weights.write_bytes(b'bad')
    with pytest.raises(ValueError,match='corrupted'):library.resolve(cp['id'])
    weights.write_bytes(payload);weights.rename(weights.with_suffix('.missing'))
    with pytest.raises(ValueError,match='missing'):library.resolve(cp['id'])
    with pytest.raises(ValueError):library.resolve('nonexistent')


def test_trash_failure_rolls_files_back_and_keeps_registry_consistent(tmp_path,monkeypatch):
    first=checkpoint(tmp_path,episode=1);second=checkpoint(tmp_path,episode=2)
    library=ModelLibrary(tmp_path);library.scan();gid=library.generations()[0]['id']
    import ai.model_library as module
    replace=module.os.replace;calls=[]
    def fail(source,destination):
        calls.append(1)
        if len(calls)==2:raise OSError('injected move failure')
        return replace(source,destination)
    monkeypatch.setattr(module.os,'replace',fail)
    with pytest.raises(OSError):library.delete(library.preview_delete('generation',gid))
    assert first.exists() and second.exists() and library.get(gid)['status']!='Deleted'


def test_path_escape_and_shared_evaluation_are_retained(tmp_path):
    path=checkpoint(tmp_path);library=ModelLibrary(tmp_path);library.scan();cp=library.find_checkpoint(path)
    outside=tmp_path/'unmanaged';outside.mkdir()
    with library.db(True) as db:db.execute('UPDATE checkpoints SET path=? WHERE id=?',(str(outside),cp['id']))
    with pytest.raises(ValueError,match='outside managed'):library.preview_delete('checkpoint',cp['id'])
    assert outside.exists()
