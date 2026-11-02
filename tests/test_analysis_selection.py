"""Independent real-checkpoint selection, single-model state and dataset provenance."""
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['SDL_AUDIODRIVER']='dummy'
import time
import pygame
import pytest

torch=pytest.importorskip('torch')
from ai.checkpoints import CheckpointStore
from ai.dqn import Trainer
from ai.evaluation import evaluate_checkpoints
from ai.history import HistoryStore
from ai.analysis_stats import EpisodeRecord
from ui.analysis_view import AnalysisView
from ui.components import reset_fonts


def settle(view):
    deadline=time.monotonic()+10
    while view.jobs and time.monotonic()<deadline:
        view.update(.016,False)
        time.sleep(.001)
    assert not view.jobs,view.status


@pytest.fixture
def selection(tmp_path):
    pygame.init();reset_fonts()
    store=CheckpointStore(tmp_path/'checkpoints')
    from ai.training_config import TrainingConfig
    trainer=Trainer(TrainingConfig(episodes=6,episode_cap=3,batch_size=2))
    journal=HistoryStore(tmp_path/'history')
    journal.register('run','Baseline DQN',trainer.online.config.sizes,{'checkpoint_run_id':'run'})
    while trainer.completed<6:
        transition=trainer.train_step()
        if transition.done:
            env=trainer.environment
            journal.append_episode(EpisodeRecord('run',trainer.completed,env.game.score,env.steps,env.total_reward,
                transition.reason,trainer.total_steps,trainer.epsilon,trainer.completed*.1,env.game.apples_eaten))
            if trainer.completed in (2,4):store.save('run','Baseline DQN',trainer)
    journal.close()
    view=AnalysisView(tmp_path)
    view.refresh();settle(view)
    yield view
    view.close();reset_fonts();pygame.quit()


def test_independent_checkpoints_same_model_and_deselect_b(selection):
    view=selection
    assert view.selected(0).episode==4 and view.selected(1) is None
    assert [r.episode for r in view.datasets[0].records]==[1,2,3,4]
    assert not any(b.action.endswith('checkpoint_b') for b in view.buttons)
    assert any(b.text=='MODEL B: None — single model' for b in view.buttons)
    view.action('model_b');settle(view)
    assert view.selected(0).model_key==view.selected(1).model_key
    assert view.selected(1).episode==4
    view.mode='raw';view.group=25
    view.text.update(start='1',end='2')
    view.action('checkpoint_b');settle(view)
    assert view.selected(0).episode==4 and view.selected(1).episode==2
    assert view.datasets[0].checkpoint_episode==4 and view.datasets[1].checkpoint_episode==2
    assert len(view.datasets[1].records)==2 and view.mode=='raw' and view.group==25
    assert (view.text['start'],view.text['end'])==('1','2')
    assert len(view.bundle['models'])==2
    view.action('export_raw');settle(view)
    path=next((view.profile/'exports').glob('*.csv'))
    assert '_vs_' in path.name and '_4ep-' in path.name and '_2ep-' in path.name
    view.action('model_b')
    assert view.datasets[1] is None  # Immediately remove stale comparison data.
    settle(view)
    assert len(view.bundle['models'])==1
    assert not any(b.action.endswith('checkpoint_b') for b in view.buttons)
    view.action('model_a');settle(view)
    assert view.selected(0) is not None


def test_training_and_real_frozen_data_are_distinct_and_checkpoint_scoped(selection):
    view=selection
    checkpoint=view.selected(0).checkpoint
    view.action('data_a')
    assert 'No frozen evaluation exists' in view.status
    assert view.datasets[0].kind=='training'
    evaluate_checkpoints([checkpoint],(7,8),view.profile/'evaluations')
    view.refresh();settle(view)
    view.action('data_a');settle(view)
    assert view.datasets[0].kind=='evaluation' and view.datasets[0].checkpoint_episode==4
    assert [r.seed for r in view.datasets[0].records]==[7,8]
    assert any('FROZEN EVALUATION' in b.text for b in view.buttons)
    view.action('checkpoint_a');settle(view)
    assert view.selected(0).episode==2 and view.datasets[0].kind=='training'
    assert not any(view.catalog[i].kind=='evaluation' for i in view.data_choices(view.selected(0)))
    view.action('data_a')
    assert 'No frozen evaluation exists' in view.status


def test_empty_catalog_and_deleted_checkpoint_clear_stale_data(selection):
    view=selection
    selected=view.selected(0)
    # Missing selected data must produce a visible error, never retain old curves.
    selected.source.joinpath('training.pt').unlink()
    view.load_selected();settle(view)
    assert view.bundle is None and view.datasets==[None,None]
    assert view.status
    view.catalog_ready(())
    assert view.indices==[None,None] and view.bundle is None
    view.action('export_raw')
    assert 'Select data' in view.status


def test_export_control_labels_and_checkpoint_options_are_actual(selection):
    view=selection
    labels={b.action:b.text for b in view.buttons}
    assert labels['train:analysis:export_raw']=='EXPORT RAW'
    assert labels['train:analysis:export_grouped']=='EXPORT GROUPED'
    options=view.checkpoints(view.selected(0).model_key)
    assert {view.catalog[i].episode for i in options}=={2,4}
    assert all(view.catalog[i].source.is_dir() for i in options)
