"""Live mode uses the ordinary event loop, viewport and existing lab controls."""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
import pygame
import pytest
torch = pytest.importorskip('torch', reason='Optional requirements-ai.txt not installed')
from app import Application
from ai.laboratory import Laboratory
from ui.neural_view import NeuralView
from ui.neural_graph import activation_color


@pytest.fixture
def app(tmp_path):
    application = Application(tmp_path / 'settings.json')
    application.lab_view = NeuralView(Laboratory(), tmp_path / 'models')
    application.screen = 'neural'
    application.rebuild_controls()
    yield application
    application.close()


def click(app, command):
    button = next(b for b in app.controls if b.action == 'lab:' + command)
    viewport = app.viewport()
    pos = (round(viewport.x + button.rect.centerx * viewport.width / 1280),
           round(viewport.y + button.rect.centery * viewport.height / 720))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos))


def test_live_controls_snapshot_pause_and_step_transition(app):
    view = app.lab_view
    click(app, 'mode:live')
    click(app, 'play_pause')
    app.update(.2)
    assert view.live.running and view.lab.simulation_steps == 1 and view.show_decision
    app.draw(.016)
    selected = view.layer, view.neuron
    state = view.lab.game.get_state()
    snapshot = view.live.decision
    click(app, 'play_pause')
    app.update(10)
    assert view.lab.game.get_state() == state and view.live.decision is snapshot
    assert (view.layer, view.neuron) == selected
    assert view.inspection() == snapshot.inspect(*selected)
    click(app, 'forward')
    assert not view.show_decision and view.lab.game.get_state() == state
    assert view.lab.previous_outputs == snapshot.outputs
    assert view.lab.outputs != snapshot.outputs
    click(app, 'last_decision')
    assert view.show_decision and view.inspection() == snapshot.inspect(*selected)
    click(app, 'stop')
    assert view.live.mode == 'step' and view.lab.game.get_state() == state
    click(app, 'step')
    assert view.lab.simulation_steps == 2


def test_speed_reset_gameover_and_replay_controls(app):
    view = app.lab_view
    click(app, 'mode:live')
    click(app, 'live_speed')
    assert view.live.speed == 10
    click(app, 'reset')
    initial = view.lab.game.get_state()
    click(app, 'play_pause')
    app.update(.1)
    assert view.lab.simulation_steps == 1
    click(app, 'reset')
    assert view.lab.game.get_state() == initial and not view.live.running
    view.lab.game.snake.body = [(24, 12), (23, 12), (22, 12)]
    with torch.no_grad():
        for p in view.lab.network.parameters(): p.zero_()
        view.lab.network.layers[-1].bias[0] = 1
    click(app, 'play_pause')
    app.update(.1)
    assert view.lab.game.terminated and not view.live.running
    app.draw(.016)
    assert any(b.text == 'RESET & PLAY AGAIN' for b in app.controls)
    assert any(b.text == 'RETURN TO STEP MODE' for b in app.controls)
    click(app, 'reset_play')
    assert view.live.running and view.lab.simulation_steps == 0


def test_tabs_focus_loss_and_returning_to_lab_pause_without_movement(app):
    view = app.lab_view
    click(app, 'mode:live')
    click(app, 'play_pause')
    app.update(.2)
    state = view.lab.game.get_state()
    click(app, 'tab:config')
    app.update(10)
    assert not view.live.running and view.lab.game.get_state() == state
    click(app, 'tab:lab')
    click(app, 'play_pause')
    app.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    app.update(10)
    assert not view.live.running and view.lab.game.get_state() == state
    click(app, 'play_pause')
    click(app, 'back')
    assert app.screen == 'ai' and not view.live.running
    app.action('open_neural')
    assert view.live.mode == 'step' and view.lab.game.get_state() == state


def test_resize_and_quit_remain_responsive_in_live_mode(app):
    click(app, 'mode:live')
    click(app, 'play_pause')
    app.window = pygame.display.set_mode((900, 900), pygame.RESIZABLE)
    app.update(.2)
    app.draw(.016)
    click(app, 'play_pause')
    assert not app.lab_view.live.running
    app.handle_event(pygame.event.Event(pygame.QUIT))
    assert not app.running


def test_activation_colors_are_fixed_signed_and_zero_is_dark():
    assert activation_color(0) != activation_color(1)
    assert activation_color(1) != activation_color(-1)
    assert activation_color(1)[1] > activation_color(.1)[1] > activation_color(0)[1]
    assert activation_color(-1)[0] > activation_color(-.1)[0]


def test_graph_connection_cache_reused_and_invalidated_by_new_trace(app):
    view = app.lab_view
    app.draw(.016)
    cached = view.connection_cache.surface
    app.draw(.016)
    assert view.connection_cache.surface is cached
    click(app, 'mode:live')
    click(app, 'play_pause')
    app.update(.2)
    app.draw(.016)
    assert view.connection_cache.surface is not cached
    cached = view.connection_cache.surface
    app.draw(.016)
    assert view.connection_cache.surface is cached


def test_editing_field_pauses_and_keeps_snapshot_until_edit_applied(app):
    view = app.lab_view
    click(app, 'mode:live')
    click(app, 'play_pause')
    app.update(.2)
    snapshot = view.live.decision
    pos = view.fields['parameter'].center
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    assert not view.live.running and view.live.decision is snapshot and view.show_decision
    app.update(2)
    assert view.lab.game.get_state() == snapshot.after
    view.text['parameter'] = '.5'
    click(app, 'edit')
    assert not view.show_decision and view.live.decision is None


def test_binary_editor_toggles_the_displayed_snapshot_observation(app):
    view = app.lab_view
    click(app, 'mode:live')
    click(app, 'play_pause')
    app.update(.2)
    snapshot = view.live.decision
    click(app, 'play_pause')
    click(app, 'forward')
    click(app, 'last_decision')
    expected = list(snapshot.observation)
    expected[3] = 1 - expected[3]
    click(app, 'input:3')
    assert view.lab.observation == tuple(expected)
    assert view.lab.previous_outputs == snapshot.outputs
    assert view.lab.input_source == 'Manual binary inputs'
    assert not view.show_decision and not view.live.running
