# Frontier experiments

Source launches with the default settings profile save Frontier experiments under
`experiments/frontier/<unique-id>/`. The Configuration screen shows the full output
path and lets you choose another directory. Custom settings profiles and packaged
launches default to a Frontier directory beside their settings instead.

Each experiment contains its configuration/identity, lightweight `models/`, full
`checkpoints/`, a versioned `history/` journal and an artifact manifest. Dashboard
**OPEN EXPERIMENT FILES** opens that exact directory. Original baseline saves are
left in their existing library.

Share a `snake_dqn_11_epXXXXXX_model.pt` together with its matching `.json`.
These files are weights-only state dictionaries with architecture, feature/action
metadata, episode, experiment identity and checksum. Load them with:

```python
from ai.frontier_store import load_published_model
network, metadata = load_published_model('experiments/frontier/<id>/models/snake_dqn_11_ep010000_model.pt')
```

For exact boundary resumption, retain the **entire checkpoint folder**, including
canonical `network.json`, weights, `training.json` and `training.pt`. Its descriptive
`snake_dqn_11_epXXXXXX_checkpoint.pt` is an alias/copy of the training-state file;
it needs the online weights and metadata beside it, and is not a standalone model.
Use the application's resume selector; model-only snapshots cannot resume training.

The repository ignore rules exclude full Frontier checkpoints, history databases
and writer locks from the default experiment path. Lightweight model artifacts
remain available for your manual Git commit/push. `tools/export_models.py` can also
copy the latest registered Frontier model into the top-level `models/` library.

See [release documentation](../docs/releases/v1.5.1.md) for retention, crash recovery,
recording guarantees and the performance measurements.
