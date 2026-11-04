"""Acceptance workflows A–G through actual application routing and temporary data."""
import os
os.environ.update(SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy')
import json
from pathlib import Path
import time
import pytest
torch=pytest.importorskip('torch');torch.set_num_threads(1)
from app import Application
from ai.model_library import ModelLibrary
from ai.checkpoints import CheckpointStore
from ai.dqn import Trainer
from ai.training_config import TrainingConfig
from ai.frontier_config import FrontierConfig


def settle(app,predicate,timeout=30):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        app.update(.016)
        if predicate():return
        time.sleep(.005)
    pytest.fail('Application workflow timed out: '+str(app.screen))


def test_existing_new_resume_analysis_delete_restore_restart(tmp_path):
    trainer=Trainer(TrainingConfig(episodes=5,episode_cap=2,batch_size=2))
    for _ in range(4):trainer.train_step()
    alpha=CheckpointStore(tmp_path/'checkpoints').save('originalalpha','Snake DQN Alpha',trainer)
    app=Application(tmp_path/'settings.json')
    try:
        # A: cold-runtime library opens without requiring a neural UI first.
        app.action('open_model_library');view=app.library_view
        alpha_cp=app.model_library.find_checkpoint(alpha)
        app.action('library:generation:'+alpha_cp['generation_id']);app.action('library:checkpoint:'+alpha_cp['id'])
        app.action('library:load')
        settle(app,lambda:app.screen=='arena')
        assert app.arena_view.arena.network.config.input_size==11
        assert app.arena_view.selected_metadata['model_name']=='Snake DQN Alpha — Gen 1'
        # B: new Distance generation, configured before any real checkpoint exists.
        app.action('open_model_library');view=app.library_view
        view.family='Distance';view.text['description']='Short acceptance run'
        app.action('library:new');assert app.screen=='frontier'
        frontier=app.frontier_view;gid=frontier.pending_generation
        frontier.text.update(episodes='3',episode_cap='3',batch_size='2')
        app.action('frontier:start')
        settle(app,lambda:frontier.snapshot and frontier.snapshot.status in ('Completed','Failed'))
        assert frontier.snapshot.status=='Completed',frontier.status
        path=Path(frontier.snapshot.checkpoint)
        assert app.model_library.find_checkpoint(path)['generation_id']==gid
        assert len(app.model_library.generations('Distance'))==1
        # C: resume keeps the generation, but creates an isolated session/history branch.
        app.action('choose_model:frontier_resume');view=app.library_view
        cp=app.model_library.find_checkpoint(path)
        app.action('library:generation:'+gid);app.action('library:checkpoint:'+cp['id']);app.action('library:load')
        frontier.text['episodes']='5';app.action('frontier:start')
        settle(app,lambda:frontier.snapshot and frontier.snapshot.status in ('Completed','Failed') and frontier.snapshot.metrics.get('completed')==5)
        assert frontier.snapshot.status=='Completed',frontier.status
        assert len(app.model_library.generations('Distance'))==1
        continued=app.model_library.find_checkpoint(frontier.snapshot.checkpoint)
        assert continued['generation_id']==gid
        # D: exact cross-family selections feed the unchanged Analysis interface.
        app.action('frontier:analysis');app.action('frontier:analysis:model_a')
        app.action('library:generation:'+alpha_cp['generation_id']);app.action('library:checkpoint:'+alpha_cp['id']);app.action('library:load')
        settle(app,lambda:not frontier.analysis.jobs)
        app.action('frontier:analysis:model_b');app.action('library:generation:'+gid)
        app.action('library:checkpoint:'+continued['id']);app.action('library:load')
        settle(app,lambda:not frontier.analysis.jobs)
        assert len([d for d in frontier.analysis.datasets if d])==2
        assert frontier.analysis.datasets[0].model_identity['encoder_id']!=frontier.analysis.datasets[1].model_identity['encoder_id']
        frontier.analysis.text['games']='2';app.action('frontier:analysis:evaluate')
        settle(app,lambda:not frontier.analysis.jobs)
        assert list((tmp_path/'evaluations').glob('*.json'))
        # E: reversible deletion only touches a temporary independent generation.
        app.action('open_model_library');app.action('library:unload')
        assert not app.library_view.status.startswith('Stop and save')
        victim=CheckpointStore(tmp_path/'checkpoints')
        first=victim.save('throwaway','Test',trainer)
        trainer.train_step();trainer.train_step();second=victim.save('throwaway','Test',trainer)
        app.library_view.refresh();library=app.model_library
        victim_cp=library.find_checkpoint(first);victim_gid=victim_cp['generation_id']
        op=library.delete(library.preview_delete('checkpoint',victim_cp['id']))
        assert not first.exists() and second.exists()
        library.restore(op)
        op=library.delete(library.preview_delete('generation',victim_gid))
        assert not first.exists() and not second.exists()
        library.restore(op);assert first.exists() and second.exists()
        # F: IDs and independent family numbering persist after a real application close.
        ids={g['id']:(g['family'],g['number']) for g in library.generations()}
    finally:app.close()
    restarted=Application(tmp_path/'settings.json')
    try:
        restarted.action('open_model_library')
        assert {g['id']:(g['family'],g['number']) for g in restarted.model_library.generations()}==ids
    finally:restarted.close()


def test_migration_follows_evidenced_legacy_resume_and_preserves_evaluations(tmp_path):
    from ai.frontier_store import FrontierStore
    trainer=Trainer(TrainingConfig(episodes=3,episode_cap=1,batch_size=2));trainer.train_step()
    original=CheckpointStore(tmp_path/'checkpoints').save('legacy','Original',trainer)
    child=FrontierStore(tmp_path/'frontier',tmp_path,trainer.config,FrontierConfig(),'Old Frontier label',identifier='continuation',resumed_from=original)
    try:
        from ai.model_registry import identity
        child.metadata.update(architecture=trainer.online.config.to_dict(),**identity(trainer.online.config))
        from ai.frontier_store import atomic_json
        atomic_json(child.root/'experiment.json',child.metadata)
        new_path,_=child.save(trainer)
    finally:child.close()
    before={str(p):p.read_bytes() for folder in (original,new_path) for p in folder.iterdir()}
    library=ModelLibrary(tmp_path);summary=library.scan()
    assert len(library.generations('Alpha'))==1
    assert library.find_checkpoint(original)['generation_id']==library.find_checkpoint(new_path)['generation_id']
    assert not summary['review']
    assert before=={str(p):p.read_bytes() for folder in (original,new_path) for p in folder.iterdir()}
