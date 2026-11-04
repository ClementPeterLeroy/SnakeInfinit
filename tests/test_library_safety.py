import json
import os
from pathlib import Path
import time
import pytest
torch=pytest.importorskip('torch');torch.set_num_threads(1)
from ai.model_library import ModelLibrary
from ai.training_config import TrainingConfig
from ai.frontier_config import FrontierConfig
from ai.frontier import FrontierWorker,FrontierHistory
from ai.history import HistoryStore
from ai.dqn import Trainer
from ai.checkpoints import CheckpointStore
from ai.persistence import ModelStore
from ai.config import NetworkConfig
from ai.network import SnakeNetwork


def saved(profile,run='run',episode=1):
    trainer=Trainer(TrainingConfig(episodes=5,episode_cap=1,batch_size=2))
    for _ in range(episode):trainer.train_step()
    return CheckpointStore(profile/'checkpoints').save(run,'Legacy',trainer)


def test_saved_model_overwrite_never_silently_substitutes_stale_identity(tmp_path):
    store=ModelStore(tmp_path/'models');path=store.save('manual',SnakeNetwork(NetworkConfig(seed=1)))
    library=ModelLibrary(tmp_path);library.scan();original=library.find_checkpoint(path)
    store.save('manual',SnakeNetwork(NetworkConfig(seed=2)),overwrite=True)
    library.scan();replacement=library.find_checkpoint(path)
    assert replacement['id']!=original['id']
    with pytest.raises(ValueError):library.resolve(original['id'])
    library.delete(library.preview_delete('generation',original['generation_id']))
    assert path.exists() and library.load(replacement['id'])[0].config.seed==2


def test_moved_checkpoint_keeps_identity_with_matching_metadata(tmp_path):
    path=saved(tmp_path);library=ModelLibrary(tmp_path);library.scan();old=library.find_checkpoint(path)
    target=tmp_path/'moved'/'checkpoints'/path.parent.name/path.name
    target.parent.mkdir(parents=True);os.replace(path,target)
    assert library.index_checkpoint(target)==old['id']
    assert library.load(old['id'])[0].config.input_size==11
    with pytest.raises(ValueError,match='outside managed'):library.preview_delete('checkpoint',old['id'])
    library.allow_root(tmp_path/'moved/checkpoints')
    assert library.preview_delete('checkpoint',old['id'])


def test_ambiguous_folder_identity_is_marked_for_review_not_grouped(tmp_path):
    path=saved(tmp_path)
    data=json.loads((path/'training.json').read_text());data['model_identifier']='someoneelse'
    (path/'training.json').write_text(json.dumps(data))
    library=ModelLibrary(tmp_path);summary=library.scan()
    assert summary['review'] and not library.checkpoints()


def test_shared_evaluation_and_history_are_retained_on_generation_deletion(tmp_path):
    from ai.evaluation import evaluate_checkpoints
    one=saved(tmp_path,'one');two=saved(tmp_path,'two')
    evaluation=evaluate_checkpoints([one,two],(1,2),tmp_path/'evaluations')
    history=HistoryStore(tmp_path/'history')
    history.register('one','One',(11,16,8,3),{'checkpoint_run_id':'one'});history.close()
    before=evaluation.read_bytes();history_bytes=(tmp_path/'history/history.sqlite3').read_bytes()
    library=ModelLibrary(tmp_path);library.scan();cp=library.find_checkpoint(one)
    preview=library.preview_delete('generation',cp['generation_id'])
    assert str(evaluation.resolve()) in preview['retained_shared']
    library.delete(preview)
    assert evaluation.read_bytes()==before and (tmp_path/'history/history.sqlite3').read_bytes()==history_bytes
    from ai.analysis_repository import AnalysisRepository
    choices=AnalysisRepository(tmp_path).choices()
    assert not any(c.model_id==cp['generation_id'] for c in choices)
    assert library.find_checkpoint(two)['status']=='Available'


def test_restore_refuses_overwrite_and_trash_path_escape(tmp_path):
    path=saved(tmp_path);library=ModelLibrary(tmp_path);library.scan();cp=library.find_checkpoint(path)
    op=library.delete(library.preview_delete('checkpoint',cp['id']))
    path.mkdir(parents=True);(path/'unrelated').write_text('keep')
    with pytest.raises(ValueError,match='overwrite'):library.restore(op)
    assert (path/'unrelated').read_text()=='keep'
    with pytest.raises(ValueError,match='Unsafe Trash'):library._trash_path(tmp_path/'outside','operation')


def test_restart_rolls_back_interrupted_prepared_deletion(tmp_path):
    path=saved(tmp_path);library=ModelLibrary(tmp_path);library.scan();cp=library.find_checkpoint(path)
    operation='interrupted';destination=library.root/'Trash'/operation/'0';destination.parent.mkdir(parents=True)
    with library.db(True) as db:
        db.execute("UPDATE checkpoints SET status='Deleting' WHERE id=?",(cp['id'],))
        db.execute('INSERT INTO trash VALUES (?,?,?,?,?,?)',(operation,'checkpoint',cp['id'],'Preparing','Available',json.dumps([(str(path),str(destination))])))
    os.replace(path,destination)
    recovered=ModelLibrary(tmp_path)
    assert path.exists() and recovered.resolve(cp['id'])


def test_permanent_sqlite_history_failure_has_readable_emergency_journal(tmp_path,monkeypatch):
    original=FrontierHistory.flush
    def fail(self):
        if self.pending:raise OSError('injected persistent SQLite failure')
        return original(self)
    monkeypatch.setattr(FrontierHistory,'flush',fail)
    worker=FrontierWorker(tmp_path/'profile',tmp_path/'experiments')
    try:
        worker.start(TrainingConfig(episodes=25,episode_cap=1,batch_size=2),FrontierConfig(environments=4))
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            result=worker.latest()
            if result and result.status=='Failed':break
            time.sleep(.005)
        worker.shutdown()
        assert result.status=='Failed'
        root=Path(result.experiment)
        assert (root/'pending-completions.json').is_file()
        records=HistoryStore(root/'history').load(root.name).records
        assert len(records)==worker.latest().metrics['completed']==25
        assert len({r.episode for r in records})==25
    finally:worker.shutdown()


def test_active_training_claim_blocks_second_learner_and_both_deletion_levels(tmp_path):
    path=saved(tmp_path);library=ModelLibrary(tmp_path);library.scan();cp=library.find_checkpoint(path)
    lease=library.acquire(cp['generation_id'],'Test trainer',training=True)
    try:
        with pytest.raises(ValueError):library.acquire(cp['generation_id'],'Other trainer',training=True)
        with pytest.raises(ValueError):library.preview_delete('checkpoint',cp['id'])
        with pytest.raises(ValueError):library.preview_delete('generation',cp['generation_id'])
    finally:library.release(lease)


def test_owned_records_move_with_generation_and_restore(tmp_path):
    from ai.evaluation import evaluate_checkpoints
    path=saved(tmp_path)
    evaluation=evaluate_checkpoints([path],(1,),tmp_path/'evaluations')
    experiment=tmp_path/'experiments'/'owned.json';experiment.parent.mkdir()
    experiment.write_text(json.dumps({'training_runs':['run']}))
    events=experiment.with_suffix('.events.jsonl');events.write_text('{}\n')
    export=tmp_path/'exports'/'owned.csv';export.parent.mkdir();export.write_text('episode,score\n1,0\n')
    sidecar=Path(str(export)+'.metadata.json')
    sidecar.write_text(json.dumps({'datasets':[{'checkpoint':str(path)}]}))
    library=ModelLibrary(tmp_path);library.scan();cp=library.find_checkpoint(path)
    owned=[evaluation,experiment,events,export,sidecar]
    payloads={str(p):p.read_bytes() for p in owned}
    preview=library.preview_delete('generation',cp['generation_id'])
    assert all(str(p.resolve()) in preview['files'] for p in owned)
    operation=library.delete(preview)
    assert all(not p.exists() for p in owned)
    library.restore(operation)
    assert payloads=={str(p):p.read_bytes() for p in owned}


def test_exported_checkpoint_copy_is_owned_and_restored_with_original(tmp_path):
    import shutil
    path=saved(tmp_path)
    copy=tmp_path/'models'/'copy';shutil.copytree(path,copy)
    # The portable export carries the original immutable checkpoint provenance.
    (copy/'training.json').unlink();(copy/'training.pt').unlink()
    (copy/'model_card.json').write_text(json.dumps({'model_identifier':'run','source_checkpoint':path.name}))
    library=ModelLibrary(tmp_path);library.scan();cp=library.find_checkpoint(path)
    assert library.find_checkpoint(copy)['id']==cp['id']
    operation=library.delete(library.preview_delete('checkpoint',cp['id']))
    assert not path.exists() and not copy.exists()
    library.restore(operation)
    assert path.exists() and copy.exists()


def test_missing_training_state_is_inference_only_and_never_resumable(tmp_path):
    path=saved(tmp_path);library=ModelLibrary(tmp_path);library.scan();cp=library.find_checkpoint(path)
    (path/'training.pt').unlink()
    assert library.checkpoints()[0]['kind']=='inference'
    with pytest.raises(ValueError,match='Training state is missing'):library.resolve(cp['id'],'resume')
    assert library.load(cp['id'])[0].config.input_size==11


def test_orphaned_paused_generation_becomes_interrupted_on_restart(tmp_path):
    library=ModelLibrary(tmp_path);gid=library.create('Distance')
    token=library.acquire(gid,'Paused worker',training=True);library.status(gid,'Paused')
    assert ModelLibrary(tmp_path).get(gid)['status']=='Paused'
    library.release(token)
    assert ModelLibrary(tmp_path).get(gid)['status']=='Interrupted'


def test_legacy_final_save_failure_remains_failed_in_library(tmp_path,monkeypatch):
    from ai.worker import TrainingWorker
    original=CheckpointStore.save;calls=[]
    def fail_final(self,*args,**kwargs):
        calls.append(1)
        if len(calls)>1:raise OSError('injected final save failure')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(CheckpointStore,'save',fail_final)
    worker=TrainingWorker(CheckpointStore(tmp_path/'checkpoints'))
    try:
        worker.start(TrainingConfig(episodes=1,episode_cap=1,batch_size=2))
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            result=worker.latest()
            if result and result.status=='Failed':break
            time.sleep(.005)
        assert result.status=='Failed'
        worker.shutdown()
        assert ModelLibrary(tmp_path).generations()[0]['status']=='Failed'
    finally:worker.shutdown()
