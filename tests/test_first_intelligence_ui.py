"""Actual Pygame configuration, worker dashboard, checkpoint and Arena routing."""
import os
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['SDL_AUDIODRIVER']='dummy'
import time
import pygame
import pytest
torch=pytest.importorskip('torch',reason='Optional requirements-ai.txt not installed')
from app import Application
from ai.laboratory import Laboratory
from ui.neural_view import NeuralView
from ui import menu


@pytest.fixture
def app(tmp_path):
    application=Application(tmp_path/'settings.json')
    application.lab_view=NeuralView(Laboratory(),tmp_path/'models')
    yield application
    application.close()


def click(app,action):
    control=next(b for b in app.controls if b.action==action)
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=control.rect.center))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP,button=1,pos=control.rect.center))


def test_mandatory_ai_menu_names_and_subtitles(app):
    app.action('ai')
    assert [(b.text,b.subtitle) for b in app.controls]==[
        ('NEURAL LABORATORY','Build & inspect'),('TRAINING CENTER','Train & monitor'),
        ('ARENA','Watch & evaluate'),('MODEL LIBRARY','Families, generations & checkpoints'),('BACK',None)]
    app.draw(.016)


def test_training_configuration_worker_charts_arena_and_lab_integration(app):
    app.action('open_training')
    view=app.training_view
    view.text.update(episodes='3',episode_cap='5',batch_size='2',checkpoint_interval='1')
    app.draw(.016)
    click(app,'train:start')
    view.worker.command('view',('headless',1))
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        app.update(.016)
        if view.snapshot and view.snapshot.status in ('Completed','Failed'):break
        time.sleep(.01)
    assert view.snapshot.status=='Completed',view.status
    app.draw(.016)
    checkpoint=view.checkpoint_path()
    assert checkpoint
    click(app,'train:arena')
    assert app.screen=='arena' and app.arena_view.arena.checkpoint_episode==3
    assert not any(p.requires_grad for p in app.arena_view.arena.network.parameters())
    app.draw(.016)
    click(app,'arena:play')
    app.update(.2)
    click(app,'arena:reset')
    assert app.arena_view.arena.environment.steps==0
    app.arena_view.text['seed']='12345'
    app.update(.016)
    click(app,'arena:reset')
    assert app.arena_view.arena.seed==12345
    click(app,'arena:select')
    app.draw(.016)
    click(app,'choose_model:arena')
    cp=app.model_library.find_checkpoint(checkpoint)
    app.action('library:generation:'+cp['generation_id'])
    app.action('library:checkpoint:'+cp['id'])
    click(app,'library:load')
    click(app,'arena:back')
    app.action('open_training')
    click(app,'train:lab')
    assert app.screen=='neural' and app.lab_view.loaded_checkpoint['episode']==3
    app.draw(.016)
    original=app.lab_view.checkpoints.load_model(checkpoint)[0].layers[0].bias.detach().clone()
    app.lab_view.lab.edit_parameter(0,0,10)
    stored=app.lab_view.checkpoints.load_model(checkpoint)[0].layers[0].bias.detach()
    assert torch.equal(original,stored)


def test_invalid_configuration_is_reported_without_starting_worker(app):
    app.action('open_training')
    app.training_view.text['gamma']='2'
    click(app,'train:start')
    assert app.training_view.worker is None and 'gamma' in app.training_view.status
    app.draw(.016)
