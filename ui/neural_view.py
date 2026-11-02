"""Interactive native Pygame lab. Torch stays behind the optional runtime boundary."""
import pygame
from ai.actions import ACTION_NAMES
from ai.config import NetworkConfig, ACTIVATIONS
from ai.observations import FEATURE_NAMES
from ai.live import LiveController, SPEEDS
from ui.components import BG, PANEL, GREEN, TEXT, MUTED, BORDER, Button, label
from ui.neural_graph import ConnectionCache, draw_network


class NeuralView:
    def __init__(self, laboratory, models_root=None):
        from ai.persistence import ModelStore
        self.lab = laboratory
        self.live = LiveController(laboratory)
        self.show_decision = False
        self.store = ModelStore(models_root)
        from ai.checkpoints import CheckpointStore
        self.checkpoints=CheckpointStore(self.store.root.parent/'checkpoints')
        self.checkpoint_index=0
        self.loaded_checkpoint=None
        self.tab = "lab"
        self.fields = {}
        self.focus = None
        self.status = "Step Mode / CPU / raw untrained outputs"
        self.pending = None
        self.layer = self.neuron = self.scroll = 0
        self.incoming = None
        self.relative = 0
        self.activation = self.lab.network.config.activation
        self.hidden = list(self.lab.network.config.hidden_sizes)
        self.text = {"seed": "0", "init_seed": "0", "neuron": "0", "parameter": "0",
                     "lr": ".001", "target0": "0", "target1": "1", "target2": "0", "name": "my-network"}
        for index, size in enumerate(self.hidden):
            self.text[f"size{index}"] = str(size)
        self.graph_hits = []
        self.connection_cache = ConnectionCache()
        self.row_hits = []
        self.selected_model = 0
        self.buttons = []
        self.parameter_text()
        self.layout()

    def add(self, title, x, y, width, action, primary=False):
        self.buttons.append(Button(title, pygame.Rect(x, y, width, 34), "lab:" + action,
                                   primary=primary, text_size=14))

    def field(self, key, x, y, width):
        self.fields[key] = pygame.Rect(x, y, width, 32)

    def layout(self):
        self.fields, self.buttons = {}, []
        if self.pending:
            self.add("CONFIRM", 480, 398, 150, "confirm", True)
            self.add("CANCEL", 650, 398, 150, "cancel")
            return
        for index, (tab, title) in enumerate((("lab", "LABORATORY"), ("config", "CONFIGURATION"),
                                              ("gradients", "GRADIENT DEMO"), ("models", "MODELS"))):
            self.add(title, 24 + index * 218, 65, 205, "tab:" + tab, self.tab == tab)
        self.add("BACK TO AI", 1100, 65, 156, "back")
        if self.tab == "lab":
            self.add("STEP MODE", 24, 112, 137, "mode:step", self.live.mode == "step")
            self.add("LIVE MODE", 168, 112, 137, "mode:live", self.live.mode == "live")
            self.field("seed", 135, 428, 170)
            if self.live.mode == "step":
                self.add("RESET ENVIRONMENT", 24, 470, 175, "reset")
                self.add("STEP", 209, 470, 96, "step")
                self.add("FORWARD PASS", 24, 511, 281, "forward")
                self.add(ACTION_NAMES[self.relative].upper(), 24, 552, 155, "relative")
                self.add("APPLY ACTION", 189, 552, 116, "apply")
                self.add("NETWORK ACTION / CHECKPOINT" if self.loaded_checkpoint else "NETWORK ACTION / UNTRAINED", 24, 593, 281, "network")
            else:
                self.add("PAUSE" if self.live.running else "PLAY", 24, 470, 175, "play_pause", True)
                self.add("STOP", 209, 470, 96, "stop")
                self.add(f"SPEED: {self.live.speed} STEPS / SECOND", 24, 511, 281, "live_speed")
                self.add("RESET & PLAY AGAIN" if self.lab.game.terminated else "RESET", 24, 552, 281,
                         "reset_play" if self.lab.game.terminated else "reset")
                self.add("RETURN TO STEP MODE" if self.lab.game.terminated else "CURRENT STATE / FORWARD",
                         24, 593, 281, "stop" if self.lab.game.terminated else "forward")
            self.add("INSPECT LAST DECISION", 24, 651, 281, "last_decision")
            for index, name in enumerate(FEATURE_NAMES):
                observation = self.live.decision.observation if self.show_decision and self.live.decision else self.lab.observation
                self.add(f"{index + 1}. {name}: {int(observation[index])}",
                         342 + index % 2 * 246, 455 + index // 2 * 31, 238, f"input:{index}")
                self.buttons[-1].rect.height = 29
            self.add("RESTORE SNAKE OBSERVATION", 342, 651, 484, "restore")
        elif self.tab == "config":
            self.add("- LAYER", 80, 190, 140, "remove")
            self.add("+ LAYER", 235, 190, 140, "add")
            for index in range(len(self.hidden)):
                self.field(f"size{index}", 260, 242 + index * 47, 115)
            self.add(self.activation.upper(), 480, 245, 300, "activation")
            self.field("init_seed", 480, 337, 300)
            self.add("BUILD NETWORK", 480, 450, 300, "build", True)
        elif self.tab == "gradients":
            for index in range(3):
                self.field(f"target{index}", 90 + index * 240, 248, 200)
            self.field("lr", 260, 330, 200)
            self.add("1. BACKPROPAGATE", 90, 402, 295, "backward", True)
            self.add("2. OPTIMIZER STEP", 410, 402, 295, "optimize")
        else:
            self.field("name", 90, 245, 620)
            self.add("SAVE CURRENT NETWORK", 90, 295, 300, "save")
            names = self.store.names()
            name = names[self.selected_model % len(names)] if names else "No saved networks"
            self.add(name, 90, 405, 620, "next_model")
            self.add("LOAD TRUSTED MODEL", 90, 455, 300, "load")
            entries=self.checkpoints.entries()
            entry=entries[self.checkpoint_index%len(entries)] if entries else None
            title=f"{entry['name']} / episode {entry['episode']}" if entry else 'No trained checkpoints'
            self.add(title,90,502,620,'next_checkpoint')
            self.add('LOAD TRUSTED CHECKPOINT',90,545,350,'load_checkpoint')
        if self.tab != "config":
            self.add(f"LAYER {self.layer} / NEXT", 860, 153, 220, "layer")
            self.field("neuron", 1150, 153, 90)
            self.add("SELECT INDEX", 1088, 195, 152, "select_neuron")
            self.add("BIAS", 860, 272, 90, "bias")
            self.field("parameter", 860, 565, 225)
            self.add("APPLY EDIT", 1095, 565, 145, "edit")

    def select(self, layer, neuron, incoming=None):
        self.layer, self.neuron, self.incoming, self.scroll = layer, neuron, incoming, 0
        self.text["neuron"] = str(neuron)
        self.parameter_text()

    def parameter_text(self):
        if self.layer < 0:
            return
        data = self.inspection()
        self.text["parameter"] = repr(data["bias"] if self.incoming is None else data["weights"][self.incoming])

    def action(self, action):
        try:
            self._action(action)
        except (ValueError, IndexError, OSError, RuntimeError) as error:
            self.status = str(error)[:150]
        self.layout()

    def inspection(self):
        if self.show_decision and self.live.decision:
            return self.live.decision.inspect(self.layer, self.neuron)
        return self.lab.network.inspect(self.lab.captured, self.layer, self.neuron)

    def pause(self, message="Live paused / decision preserved"):
        self.live.pause(message)
        self.status = message
        self.layout()

    def update(self, dt):
        if self.tab != "lab" or self.pending:
            self.live.pause("Live paused while inspecting another tab")
            return
        was_running = self.live.running
        try:
            steps = self.live.advance(dt)
        except (ValueError, RuntimeError) as error:
            self.pause(f"Live inference stopped: {error}")
            return
        if steps:
            self.show_decision = True
            self.status = self.live.message
        if steps or was_running != self.live.running:
            self.status = self.live.message
            self.layout()

    def selected_seed(self):
        return int(self.text["seed"]) if self.text["seed"].strip() else None

    def load_checkpoint(self,path):
        model,metadata=self.checkpoints.load_model(path)
        if not isinstance(model.config,NetworkConfig):raise ValueError('Use Frontier and Arena for registered models; neuron editing supports the legacy 11-input MLP')
        self.live.stop()
        self.lab.replace_network(model)
        self.loaded_checkpoint=metadata
        self.live.model_label=f"CHECKPOINT EPISODE {metadata['episode']}"
        self.live.decision=self.live.previous_decision=None
        self.show_decision=False
        self.hidden=list(model.config.hidden_sizes)
        self.activation=model.config.activation
        self.text['init_seed']='' if model.config.seed is None else str(model.config.seed)
        for index,size in enumerate(self.hidden):
            self.text[f'size{index}']=str(size)
        self.select(0,0)
        self.status=f"Loaded {metadata['model_name']} / episode {metadata['episode']} / local experiment copy"
        self.layout()

    def _action(self, action):
        if action.startswith("mode:"):
            self.live.set_mode(action.split(":")[1])
            self.status = self.live.message
            return
        if action == "play_pause":
            if self.live.running:
                self.live.pause()
            else:
                self.live.play()
            self.status = self.live.message
            return
        if action == "stop":
            self.live.stop()
            self.status = self.live.message
            return
        if action == "live_speed":
            self.live.set_speed(SPEEDS[(SPEEDS.index(self.live.speed) + 1) % len(SPEEDS)])
            self.status = self.live.message
            return
        if action == "last_decision":
            self.pause()
            if self.live.decision is None:
                raise ValueError("No live decision yet; select LIVE MODE and PLAY")
            self.show_decision = True
            self.parameter_text()
            self.status = "PRE-ACTION decision snapshot / board shows the current post-action state"
            return
        # Parameter/input changes and manual steps are explicit inspection actions.
        # Pause first so the displayed calculation cannot change under the editor.
        changes_trace = (action in ("reset", "reset_play", "step", "forward", "apply", "network", "restore",
                                    "edit", "backward", "optimize", "build", "load")
                         or action.startswith("input:")
                         or action == "confirm" and self.pending and self.pending[0] != "save")
        was_snapshot = self.show_decision
        if changes_trace:
            self.live.pause()
            if self.show_decision:
                if action.startswith("input:"):
                    # Toggle the observation the user actually sees, not the newer board's input.
                    snapshot = self.live.decision
                    self.lab.observation, self.lab.captured, self.lab.outputs = (
                        snapshot.observation, snapshot.captured, snapshot.outputs)
                elif action not in ("forward", "restore", "reset", "reset_play", "step", "apply"):
                    self.lab.snake_inputs()
            self.show_decision = False
        if action == "cancel":
            self.pending = None
            return
        if action == "confirm":
            operation, value = self.pending
            self.pending = None
            if operation == "build":
                self.lab.build(value)
                self.loaded_checkpoint=None
                self.live.model_label='UNTRAINED'
            elif operation == "save":
                self.store.save(value, self.lab.network, overwrite=True, demo_steps=self.lab.demo_steps)
            elif operation=='trained':
                self.load_checkpoint(value)
            else:
                network, metadata = self.store.load(value)
                if not isinstance(network.config,NetworkConfig):raise ValueError('Registered models use Frontier and Arena; legacy neuron editing requires 11 inputs')
                self.lab.replace_network(network)
                self.loaded_checkpoint=None
                self.live.model_label='UNTRAINED'
                self.lab.demo_steps = metadata.get("supervised_demo_steps", 0)
                self.hidden = list(network.config.hidden_sizes)
                self.activation = network.config.activation
                self.text["init_seed"] = "" if network.config.seed is None else str(network.config.seed)
                for index, size in enumerate(self.hidden):
                    self.text[f"size{index}"] = str(size)
            self.select(0, 0)
            if operation != "save":
                self.live.decision = self.live.previous_decision = None
            self.status = f"{operation.capitalize()} complete"
        elif action.startswith("tab:"):
            self.live.pause("Live paused / tab changed")
            self.status = self.live.message
            self.tab, self.focus = action.split(":")[1], None
            pygame.key.stop_text_input()
        elif action in ("reset", "reset_play"):
            self.live.reset(self.selected_seed())
            if action == "reset_play":
                self.live.play()
            self.status = self.live.message
        elif action == "step":
            self.lab.step()
        elif action == "forward":
            if self.live.mode == "live" or was_snapshot:
                self.lab.snake_inputs()
            else:
                self.lab.forward()
            self.status = "Current post-action environment / fresh forward pass (no movement)"
        elif action == "relative":
            self.relative = (self.relative + 1) % 3
        elif action == "apply":
            self.lab.step(self.relative)
        elif action == "network":
            selected = self.lab.network_action()
            source='checkpoint' if self.loaded_checkpoint else 'UNTRAINED'
            self.status = f"One {source} action: {ACTION_NAMES[selected]} (argmax of current inputs)"
        elif action.startswith("input:"):
            index = int(action.split(":")[1])
            self.lab.set_input(index, 1 - self.lab.observation[index])
        elif action == "restore":
            self.lab.snake_inputs()
        elif action in ("add", "remove"):
            if action == "add" and len(self.hidden) < 5:
                self.text[f"size{len(self.hidden)}"] = "16"
                self.hidden.append(16)
            elif action == "remove" and len(self.hidden) > 1:
                self.hidden.pop()
        elif action == "activation":
            self.activation = ACTIVATIONS[(ACTIVATIONS.index(self.activation) + 1) % len(ACTIVATIONS)]
        elif action == "build":
            config = NetworkConfig(tuple(int(self.text[f"size{i}"]) for i in range(len(self.hidden))),
                                   self.activation, int(self.text["init_seed"]) if self.text["init_seed"].strip() else None)
            self.pending = ("build", config)
        elif action == "layer":
            self.select((self.layer + 1) % len(self.lab.network.layers), 0)
        elif action == "select_neuron":
            neuron = int(self.text["neuron"])
            if self.layer < 0:
                if not 0 <= neuron < 11:
                    raise ValueError("Input neuron index must be 0..10")
            else:
                self.lab.network.inspect(self.lab.captured, self.layer, neuron)
            self.select(self.layer, neuron)
        elif action == "bias":
            self.incoming = None
            self.parameter_text()
        elif action == "edit":
            if self.layer < 0:
                raise ValueError("Use the binary input controls to edit inputs")
            self.lab.edit_parameter(self.layer, self.neuron, float(self.text["parameter"]), self.incoming)
            self.live.decision = self.live.previous_decision = None
            self.status = "Parameter manually edited; forward pass refreshed; Adam state reset"
        elif action == "backward":
            self.lab.backward([self.text[f"target{i}"] for i in range(3)], self.text["lr"])
            self.status = "Gradients ready: inspect a weight or bias, then choose OPTIMIZER STEP"
        elif action == "optimize":
            self.lab.optimizer_step()
            self.live.decision = self.live.previous_decision = None
            self.parameter_text()
            self.status = "One Adam update complete; inspector retains the last backward gradients"
        elif action == "save":
            name = self.text["name"].strip()
            path = self.store.path(name)
            if path.exists():
                self.pending = ("save", name)
            else:
                self.store.save(name, self.lab.network, demo_steps=self.lab.demo_steps)
                self.status = "Network saved"
        elif action == "next_model":
            self.selected_model += 1
        elif action=='next_checkpoint':
            self.checkpoint_index+=1
        elif action=='load_checkpoint':
            entries=self.checkpoints.entries()
            if not entries:
                raise ValueError('No trained checkpoints; create a run in Training Center')
            self.pending=('trained',entries[self.checkpoint_index%len(entries)]['path'])
        elif action == "load":
            names = self.store.names()
            if not names:
                raise ValueError("No saved models")
            self.pending = ("load", names[self.selected_model % len(names)])

    def handle_event(self, event, mouse):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.pending:
                self.action("cancel")
                return True
            if self.focus:
                self.focus = None
                pygame.key.stop_text_input()
                return True
            return False
        if self.pending:
            return event.type in (pygame.TEXTINPUT, pygame.KEYDOWN, pygame.MOUSEWHEEL)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.focus = next((key for key, rect in self.fields.items() if rect.collidepoint(mouse)), None)
            if self.focus:
                if self.live.running:
                    self.pause("Live paused / editing field")
                pygame.key.start_text_input()
                return True
            pygame.key.stop_text_input()
            if self.tab == "lab":
                for rect, layer, start, end in self.graph_hits:
                    if rect.collidepoint(mouse):
                        self.select(layer, start)
                        self.layout()
                        return True
            for rect, incoming in self.row_hits:
                if rect.collidepoint(mouse):
                    self.incoming = incoming
                    self.parameter_text()
                    return True
        if event.type == pygame.MOUSEWHEEL and self.layer >= 0 and mouse[0] >= 850:
            maximum = max(0, self.lab.network.layers[self.layer].in_features - 8)
            self.scroll = max(0, min(maximum, self.scroll - event.y))
            return True
        if self.focus:
            if event.type == pygame.TEXTINPUT:
                self.text[self.focus] = (self.text[self.focus] + "".join(c for c in event.text if c.isprintable()))[:40]
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.text[self.focus] = self.text[self.focus][:-1]
                elif event.key == pygame.K_RETURN:
                    self.focus = None
                    pygame.key.stop_text_input()
                elif event.key == pygame.K_a and getattr(event, "mod", 0) & pygame.KMOD_CTRL:
                    self.text[self.focus] = ""
                return True
        return False

    def draw(self, surface):
        surface.fill(BG)
        label(surface, "NEURAL LABORATORY / Build & inspect", (24, 18), 24, GREEN, False)
        if self.loaded_checkpoint:
            label(surface,f"{self.loaded_checkpoint['model_name']} / episode {self.loaded_checkpoint['episode']}",(650,25),16,MUTED,False)
        label(surface, self.status[:145], (24, 699), 13, MUTED, False)
        self.row_hits = []
        if self.tab == "lab":
            self.draw_board(surface)
            snapshot = self.live.decision if self.show_decision else None
            self.graph_hits = draw_network(surface, pygame.Rect(342, 119, 484, 215), self.lab,
                                          (self.layer, self.neuron), snapshot=snapshot,
                                          connection_cache=self.connection_cache)
            self.draw_outputs(surface, 342, 372, decision_panel=True)
            source = f"LAST DECISION #{snapshot.step} / PRE-ACTION inputs" if snapshot else self.lab.input_source
            label(surface, source, (342, 430), 14, MUTED, False)
            label(surface, "Reset seed:", (24, 435), 15, MUTED, False)
        elif self.tab == "config":
            label(surface, "Input: 11     Output: 3     CPU / float32", (80, 130), 22, TEXT, False)
            label(surface, f"Hidden layers: {len(self.hidden)} (1..5)", (80, 155), 18, MUTED, False)
            for index in range(len(self.hidden)):
                label(surface, f"Hidden {index + 1} (2..128):", (80, 248 + index * 47), 18, TEXT, False)
            label(surface, "Hidden activation / linear output", (480, 210), 18, MUTED, False)
            label(surface, "Initialization seed (blank = random)", (480, 302), 18, MUTED, False)
            label(surface, f"Actual model: {self.lab.network.parameter_count:,} trainable parameters", (80, 545), 22, GREEN, False)
            label(surface, "Current: " + " > ".join(map(str, self.lab.network.config.sizes)), (80, 585), 20, TEXT, False)
            label(surface, "Building replaces all existing weights and optimizer state.", (80, 630), 18, MUTED, False)
        elif self.tab == "gradients":
            label(surface, "Supervised MSE example / Adam / no DQN", (90, 130), 24, GREEN, False)
            label(surface, "Targets are educational values, not optimal Snake actions.", (90, 175), 18, MUTED, False)
            for index, name in enumerate(ACTION_NAMES):
                label(surface, name + " target", (90 + index * 240, 215), 18, TEXT, False)
            label(surface, "Learning rate (0..1]:", (90, 335), 18, TEXT, False)
            before, after = self.lab.loss_before, self.lab.loss_after
            label(surface, f"MSE before: {before:.8g}" if before is not None else "MSE before: --", (90, 468), 22, TEXT, False)
            label(surface, f"MSE after: {after:.8g}" if after is not None else "MSE after: --", (90, 508), 22, TEXT, False)
            self.draw_outputs(surface, 90, 560)
            label(surface, f"Optimizer updates: {self.lab.demo_steps} / gradients reset each backward", (90, 625), 17, MUTED, False)
        else:
            label(surface, "Local model library", (90, 135), 26, GREEN, False)
            label(surface, "Name: letters, digits, hyphens and underscores (max 40)", (90, 207), 18, MUTED, False)
            label(surface, "JSON architecture + weights-only PyTorch state dictionary", (90, 350), 17, TEXT, False)
            label(surface, "Load only trusted models. Experiments never alter saved checkpoints.", (90, 172), 16, MUTED, False)
            path = str(self.store.root)
            label(surface, "Writable model directory:", (90, 595), 16, MUTED, False)
            for index in range(0, len(path), 72):
                label(surface, path[index:index + 72], (90, 622 + index // 72 * 22), 14, TEXT, False)
        if self.tab != "config":
            self.draw_inspector(surface)
        for key, rect in self.fields.items():
            pygame.draw.rect(surface, PANEL, rect, border_radius=5)
            pygame.draw.rect(surface, GREEN if self.focus == key else BORDER, rect, 1, border_radius=5)
            rendered = self.text.get(key, "")
            # Clip long values within the numeric field; underlying text remains exact.
            clip = surface.get_clip()
            surface.set_clip(rect.inflate(-8, -2))
            label(surface, rendered, (rect.x + 6, rect.y + 5), 16, TEXT, False)
            surface.set_clip(clip)
        if self.pending:
            shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 190))
            surface.blit(shade, (0, 0))
            pygame.draw.rect(surface, PANEL, (330, 265, 620, 200), border_radius=12)
            operation = self.pending[0]
            title = {"build": "Replace current network weights?", "save": "Overwrite existing saved model?",
                     "load": "Load trusted local model and replace weights?",'trained':'Load trusted checkpoint into an experiment copy?'}[operation]
            label(surface, title, (640, 320), 22)
            label(surface, "This action replaces the corresponding parameters.", (640, 355), 16, MUTED)

    def draw_outputs(self, surface, x, y, decision_panel=False):
        snapshot = self.live.decision if self.show_decision else None
        outputs = snapshot.outputs if snapshot else self.lab.outputs
        selected = max(range(3), key=lambda index: outputs[index])
        if decision_panel:
            label(surface, "NETWORK DECISION / RAW Q-VALUES" if self.loaded_checkpoint else "NETWORK DECISION / RAW UNTRAINED OUTPUTS", (x, y - 19), 13, MUTED, False)
        for index, name in enumerate(("Straight", "Left", "Right")):
            value = outputs[index]
            previous = self.live.previous_decision
            old = (previous.outputs if previous else None) if snapshot else self.lab.previous_outputs
            if decision_panel and index == selected:
                pygame.draw.rect(surface, PANEL, (x + index * 162 - 3, y - 2, 157, 21), border_radius=3)
            formatted = f"{value:+.4f}" if abs(value) < 1e4 else f"{value:+.4g}"
            label(surface, f"{name}: {formatted}", (x + index * 162, y), 14,
                  GREEN if index == selected else MUTED, False)
            delta = f"change {value - old[index]:+.4g}" if old is not None else "change --"
            label(surface, delta, (x + index * 162, y + 16), 11, MUTED, False)
        if decision_panel:
            label(surface, "SELECTED ACTION: " + ACTION_NAMES[selected].upper(), (x, y + 36), 15, GREEN, False)

    def draw_board(self, surface):
        state = self.lab.game.get_state()
        board = pygame.Rect(54, 155, 220, 220)
        pygame.draw.rect(surface, PANEL, board, border_radius=5)
        cell = board.width / state.board_width
        for index, (x, y) in enumerate(state.snake):
            rect = pygame.Rect(board.x + x * cell + 1, board.y + y * cell + 1, cell - 1, cell - 1)
            pygame.draw.rect(surface, GREEN if index == 0 else (64, 148, 98), rect, border_radius=2)
        if state.apple is not None:
            x, y = state.apple
            pygame.draw.circle(surface, (229, 124, 119), (round(board.x + (x + .5) * cell), round(board.y + (y + .5) * cell)), 4)
        if state.terminated:
            label(surface, f"GAME OVER / score {state.score} / steps {self.lab.simulation_steps}", (24, 383), 15, GREEN, False)
            label(surface, "Cause: " + str(state.death_cause or state.status.value), (24, 406), 14, MUTED, False)
        else:
            label(surface, f"Score {state.score} / steps {self.lab.simulation_steps} / {state.status.value}", (24, 383), 16, TEXT, False)
            board_label = "POST-ACTION BOARD" if self.show_decision else "CURRENT CLASSIC ENVIRONMENT"
            label(surface, board_label, (24, 406), 13, MUTED, False)

    def draw_inspector(self, surface):
        pygame.draw.line(surface, BORDER, (842, 118), (842, 682))
        label(surface, "NEURON INSPECTOR (indices start at 0)", (860, 119), 16, GREEN, False)
        source = f"LAST DECISION #{self.live.decision.step} / PRE-ACTION" if self.show_decision and self.live.decision else "CURRENT FORWARD CAPTURE"
        label(surface, source, (860, 139), 10, MUTED, False)
        label(surface, "Neuron", (1090, 160), 13, MUTED, False)
        if self.layer < 0:
            observation = self.live.decision.observation if self.show_decision and self.live.decision else self.lab.observation
            label(surface, f"Input {self.neuron}: {observation[self.neuron]:.0f}", (860, 240), 20, TEXT, False)
            label(surface, FEATURE_NAMES[self.neuron], (860, 280), 20, MUTED, False)
            return
        data = self.inspection()
        label(surface, f"z = sum(w*x) + b = {data['weighted_sum']:.7g}", (860, 237), 17, TEXT, False)
        edited = self.lab.network.edited_parameters
        bias_mark = " * edited" if (self.layer, self.neuron, None) in edited else ""
        label(surface, f"{data['bias']:.7g}{bias_mark}", (963, 280), 16, GREEN if self.incoming is None else MUTED, False)
        for row, incoming in enumerate(range(self.scroll, min(self.scroll + 8, len(data['inputs'])))):
            x, weight = data['inputs'][incoming], data['weights'][incoming]
            rect = pygame.Rect(856, 313 + row * 26, 390, 25)
            if self.incoming == incoming:
                pygame.draw.rect(surface, PANEL, rect, border_radius=3)
            marker = " *" if (self.layer, self.neuron, incoming) in edited else ""
            label(surface, f"{incoming}: {x:.4g} x {weight:.5g} = {x * weight:.5g}{marker}", (860, rect.y + 2), 15, TEXT, False)
            self.row_hits.append((rect, incoming))
        label(surface, f"Scroll inputs {self.scroll}..{min(self.scroll + 7, len(data['inputs']) - 1)} / {len(data['inputs'])}", (860, 525), 13, MUTED, False)
        label(surface, f"{data['activation']}: a = {data['output']:.7g}", (860, 546), 15, GREEN, False)
        label(surface, "Editing bias" if self.incoming is None else f"Editing weight {self.incoming}", (860, 609), 15, MUTED, False)
        gradient = self.lab.gradient(self.layer, self.neuron, self.incoming)
        label(surface, f"Last backward gradient: {gradient:.7g}" if gradient is not None else "Last backward gradient: --", (860, 639), 16, TEXT, False)
        label(surface, "* = manually edited / click a row to edit its weight", (860, 667), 13, MUTED, False)
