"""Exercise native lab controls with actual Torch and an offscreen SDL display."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
import time
import pygame
import pytest
torch = pytest.importorskip("torch", reason="Optional requirements-ai.txt not installed")
from app import Application
from ai.laboratory import Laboratory
from ai.config import NetworkConfig
from ui.neural_view import NeuralView


@pytest.fixture
def app(tmp_path):
    application = Application(tmp_path / "settings.json")
    application.lab_view = NeuralView(Laboratory(), tmp_path / "models")
    application.screen = "neural"
    application.rebuild_controls()
    yield application
    application.close()


def click(app, action):
    control = next(button for button in app.controls if button.action == "lab:" + action)
    pos = control.rect.center
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=pos))


def test_neural_ui_environment_manual_inputs_and_inspector(app):
    app.draw(.016)
    lab, view = app.lab_view.lab, app.lab_view
    original = lab.game.get_state()
    app.update(1000)
    assert lab.game.get_state() == original and app.game is None
    click(app, "step")
    assert lab.game.elapsed == pytest.approx(original.interval)
    click(app, "reset")
    assert lab.game.get_state() == original
    click(app, "forward")
    assert lab.game.get_state() == original
    click(app, "input:0")
    assert lab.input_source == "Manual binary inputs"
    click(app, "restore")
    assert lab.input_source == "Snake snapshot"
    click(app, "relative")
    click(app, "apply")
    click(app, "network")
    app.draw(.016)
    rect, incoming = view.row_hits[0]
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center))
    assert view.incoming == incoming
    before = lab.outputs
    view.text['parameter'] = "0.75"
    click(app, "edit")
    assert (view.layer, view.neuron, incoming) in lab.network.edited_parameters
    assert lab.previous_outputs == before
    view.text['parameter'] = "nan"
    click(app, "edit")
    assert "finite" in view.status


def test_configuration_build_confirmation_and_all_layer_inputs(app):
    view = app.lab_view
    original = view.lab.network
    click(app, "tab:config")
    view.text['size0'] = '129'
    click(app, 'build')
    assert view.pending is None and view.lab.network is original
    view.text['size0'] = '128'
    for _ in range(3):
        click(app, 'add')
    assert len(view.hidden) == 5
    for index in range(5):
        view.text[f'size{index}'] = '128'
    click(app, 'activation')
    click(app, 'build')
    assert view.pending and view.lab.network is original
    app.draw(.016)
    click(app, 'cancel')
    assert view.lab.network is original
    click(app, 'build')
    click(app, 'confirm')
    assert view.lab.network.config.hidden_sizes == (128,) * 5
    click(app, 'tab:lab')
    app.draw(.016)
    assert len(view.graph_hits) <= 16 * 7
    view.select(1, 127)
    view.layout()
    app.rebuild_controls()
    assert len(view.lab.network.inspect(view.lab.captured, 1, 127)['weights']) == 128
    view.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, y=-200), (950, 400))
    assert view.scroll == 120
    app.draw(.016)
    assert view.row_hits[-1][1] == 127


def test_supervised_demo_and_persistence_through_ui(app):
    view, lab = app.lab_view, app.lab_view.lab
    click(app, 'tab:gradients')
    click(app, 'backward')
    assert lab.gradient_ready
    app.draw(.016)
    click(app, 'optimize')
    assert lab.demo_steps == 1 and lab.loss_after < lab.loss_before
    click(app, 'tab:models')
    click(app, 'save')
    assert view.store.names() == ['my-network']
    click(app, 'save')
    assert view.pending == ('save', 'my-network')
    click(app, 'confirm')
    original = lab.outputs
    lab.edit_parameter(2, 0, 100)
    click(app, 'load')
    assert view.pending == ('load', 'my-network') and lab.outputs != original
    click(app, 'confirm')
    assert lab.outputs == original and lab.demo_steps == 1
    app.draw(.016)


def test_text_editing_focus_and_escape(app):
    view = app.lab_view
    pos = view.fields['seed'].center
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    assert view.focus == 'seed'
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a, mod=pygame.KMOD_CTRL))
    app.handle_event(pygame.event.Event(pygame.TEXTINPUT, text='1234'))
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
    assert view.text['seed'] == '1234' and view.focus is None
    click(app, 'reset')
    assert view.lab.environment_seed == 1234
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    assert app.screen == 'ai'


def test_optional_runtime_loaded_asynchronously(tmp_path):
    application = Application(tmp_path / 'settings.json')
    try:
        application.action('open_neural')
        assert application.screen == 'neural_loading'
        deadline = time.monotonic() + 30
        while application.screen == 'neural_loading' and time.monotonic() < deadline:
            application.update(.016)
            time.sleep(.01)
        assert application.screen == 'neural', application.lab_error
        application.draw(.016)
        assert application.lab_view.lab.network.parameter_count == 355
    finally:
        application.close()


def test_broken_optional_runtime_does_not_crash(app):
    app.lab_view = None
    app.lab_thread = object()  # a worker is already pending
    app.lab_queue.put((False, 'Missing native library'))
    app.screen = 'neural_loading'
    app.update(0)
    assert app.screen == 'neural_unavailable'
    app.draw(.016)
    app.action('play:arcade')
    assert app.game.rules.mode == 'arcade'


