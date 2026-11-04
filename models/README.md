# Shared SnakeInfinite models

This visible repository folder contains model weights and network configurations
intended to be included in GitHub commits. Nothing here is excluded by `.gitignore`.

Refresh it from your local trained models after training:

```powershell
.\.venv\Scripts\python.exe tools\export_models.py
```

The command copies the latest completed checkpoint for each training run and saved
Laboratory models. It leaves the original files unchanged and does not run Git.
Each training export includes:

- `network.json`: architecture, input/action definitions, weights filename and checksum.
- `weights-....pt`: original PyTorch network weights.
- `model_card.json`: training episode, step count, configuration and provenance.

These are portable inference models. The export tool supports historical 11-input
models and v1.5.2 registered distance/spatial models. Keep architecture/encoder
metadata with its matching weights.

Frontier also publishes shareable model pairs directly under
`experiments/frontier/<experiment-id>/models/`, with descriptive prefixes for
Legacy11, Distance19, Binary CNN and Ordered CNN. Those `.pt` and matching `.json`
files include episode, training-step count, experiment and model identities.
v1.5.3 model metadata also records collection configuration, actual optimizer
updates, active training duration and session wall duration. Frozen inference
uses the saved network/encoder independently of training environment count.
Use **OPEN EXPERIMENT FILES** to find them. See
[artifact formats](../docs/spatial-architecture.md#artifacts-and-compatibility).

Full resumable checkpoints, replay memory, optimizer state and live training
histories stay in the run's checkpoint/history directories (Frontier) or the local
user-data library (Legacy). The lightweight exports do not resume training.
Export folders are snapshots; run the command again to add a later episode. An
existing identical export is reused; different existing contents are never overwritten.

With optional AI dependencies installed, load a repository model in Python:

```python
from pathlib import Path
from ai.persistence import ModelStore

store = ModelStore(Path('models'))
print(store.names())
network, metadata = store.load(store.names()[0])
```

Stage the desired model folder alongside your code in VS Code, then commit and
push manually. Keep its weights and `network.json` together when sharing.
