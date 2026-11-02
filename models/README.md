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

These are portable inference models. Full resumable checkpoints, replay memory,
optimizer state and live training histories remain in the local user-data library.
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
