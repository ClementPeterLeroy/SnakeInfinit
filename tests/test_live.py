"""Real inference, immutable decision snapshots, and deterministic live scheduling."""
from dataclasses import FrozenInstanceError
import pytest
torch = pytest.importorskip("torch", reason="Optional requirements-ai.txt not installed")
from ai.actions import RelativeAction
from ai.laboratory import Laboratory
from ai.live import LiveController, SPEEDS
from ai.observations import encode_observation
from core.types import Direction


def controller(action=1):
    lab = Laboratory(environment_seed=123)
    # A deterministic real network. Left turns make a safe four-cell circuit.
    with torch.no_grad():
        for parameter in lab.network.parameters():
            parameter.zero_()
        lab.network.layers[-1].bias[action] = 1
    lab.game.apple = (0, 0)
    lab.snake_inputs()
    return LiveController(lab)


def test_live_starts_from_current_state_and_infers_before_each_movement(monkeypatch):
    live = controller()
    assert live.mode == "step" and not live.running and live.speed == 5
    before = live.lab.game.get_state()
    live.lab.set_input(0, 1)  # live must ignore manual observations
    calls = []
    capture = live.lab.network.capture
    def tracked(observation):
        calls.append((live.lab.game.get_state(), observation))
        return capture(observation)
    monkeypatch.setattr(live.lab.network, "capture", tracked)
    live.play()
    assert live.advance(.2) == 1
    assert len(calls) == 1 and calls[0] == (before, encode_observation(before))
    assert live.lab.game.snake.direction == Direction.UP
    assert live.lab.simulation_steps == 1 and live.decision.action == RelativeAction.LEFT
    assert live.decision.after == live.lab.game.get_state()


@pytest.mark.parametrize("speed", SPEEDS)
@pytest.mark.parametrize("fps", (30, 60, 120))
def test_simulation_speed_independent_of_render_fps(speed, fps):
    live = controller()
    live.set_speed(speed)
    live.play()
    for _ in range(fps * 2):
        live.advance(1 / fps)
    assert live.lab.simulation_steps == speed * 2
    assert live.accumulator == pytest.approx(0, abs=1e-8)


def test_pause_resume_stop_and_mode_switch_preserve_everything():
    live = controller()
    live.play()
    live.advance(.35)
    state, observation, outputs = live.lab.game.get_state(), live.lab.observation, live.lab.outputs
    parameters = [p.detach().clone() for p in live.lab.network.parameters()]
    snapshot = live.decision
    live.pause()
    assert live.advance(100) == 0
    assert live.lab.game.get_state() == state
    assert (live.lab.observation, live.lab.outputs) == (observation, outputs)
    assert all(torch.equal(a, b) for a, b in zip(parameters, live.lab.network.parameters()))
    assert live.decision is snapshot and live.accumulator == 0
    live.play()
    assert live.advance(.19) == 0
    assert live.advance(.01) == 1
    state = live.lab.game.get_state()
    live.stop()
    assert live.mode == "step" and not live.running
    live.set_mode("live")
    assert live.lab.game.get_state() == state and not live.running


def test_seeded_reset_preserves_network_but_clears_session():
    live = controller()
    original = live.reset(21)
    parameters = [p.detach().clone() for p in live.lab.network.parameters()]
    live.play()
    live.advance(.4)
    assert live.reset(21) == original
    assert not live.running and live.decision is None and live.previous_decision is None
    assert live.lab.simulation_steps == 0
    assert all(torch.equal(a, b) for a, b in zip(parameters, live.lab.network.parameters()))


def test_snapshot_captures_real_pre_action_values_and_survives_future_edits():
    lab = Laboratory(environment_seed=21)
    live = LiveController(lab)
    before = lab.game.get_state()
    expected_observation = encode_observation(before)
    expected_capture = lab.network.capture(expected_observation)
    live.play()
    live.advance(.2)
    snapshot = live.decision
    assert snapshot.before == before
    assert snapshot.after != snapshot.before and snapshot.after == lab.game.get_state()
    assert snapshot.observation == expected_observation and snapshot.captured == expected_capture
    assert snapshot.outputs == expected_capture[-1].outputs
    assert snapshot.action == max(range(3), key=lambda i: snapshot.outputs[i])
    inspection = snapshot.inspect(0, 0)
    assert inspection['weighted_sum'] == pytest.approx(sum(x * w for x, w in zip(inspection['inputs'], inspection['weights'])) + inspection['bias'])
    old_bias = inspection['bias']
    live.pause()
    lab.snake_inputs()
    lab.edit_parameter(0, 0, 12)
    assert snapshot.inspect(0, 0)['bias'] == old_bias
    assert snapshot.outputs == expected_capture[-1].outputs
    with pytest.raises(FrozenInstanceError):
        snapshot.action = RelativeAction.RIGHT


def test_decision_tracks_changing_observation_not_previous_manual_forward():
    live = controller()
    live.lab.set_input(10, 1)
    live.play()
    live.advance(.2)
    first = live.decision
    live.advance(.2)
    second = live.decision
    assert second.before == first.after
    assert second.observation == encode_observation(second.before)
    assert second.observation != first.observation  # direction changed
    assert live.previous_decision is first
    assert second.captured == live.lab.network.capture(second.observation)


def test_terminal_stops_without_automatic_restart_and_preserves_final_decision():
    live = controller(action=0)
    live.set_speed(20)
    live.play()
    for _ in range(100):
        live.advance(.05)
    assert not live.running and live.lab.game.terminated
    assert live.lab.simulation_steps == 13
    assert live.decision.after.terminated and live.decision.after.death_cause
    final = live.lab.game.get_state()
    snapshot = live.decision
    live.play()
    assert not live.running and live.advance(2) == 0
    assert live.decision is snapshot and live.lab.game.get_state() == final


def test_bounded_catch_up_preserves_backlog_and_reports_excessive_stalls():
    live = controller()
    live.set_speed(20)
    live.play()
    assert live.advance(1) == live.MAX_STEPS_PER_FRAME
    assert live.accumulator == pytest.approx(.6)
    assert live.advance(0) == 8
    assert live.advance(0) == 4
    assert live.lab.simulation_steps == 20
    assert live.accumulator == pytest.approx(0)
    assert live.advance(6) == 0
    assert not live.running and "6.00s backlog canceled" in live.message
    live.play()
    assert live.advance(0) == 0  # explicit resume starts a fresh clock


def test_speed_changes_clear_fractional_timing_without_resetting_board():
    live = controller()
    live.play()
    live.advance(.1)
    state = live.lab.game.get_state()
    live.set_speed(10)
    assert live.lab.game.get_state() == state
    assert live.advance(.09) == 0
    assert live.advance(.01) == 1
    with pytest.raises(ValueError):
        live.set_speed(7)


def test_inference_uses_custom_current_state_without_resetting_it():
    live = controller(2)
    live.lab.game.snake.body = [(5, 5), (4, 5), (3, 5)]
    live.lab.game.apple = (5, 7)
    state = live.lab.game.get_state()
    live.play()
    live.advance(.2)
    assert live.decision.before == state
    assert live.decision.observation[8] == 1
    assert live.lab.game.snake.body[0] == (5, 6)
