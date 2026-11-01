"""Bounded, validated JSON + weights-only state dictionaries in writable profiles."""
from datetime import datetime, timezone
import hashlib
import json
import pickle
from pathlib import Path
import re
from uuid import uuid4
import zipfile
import torch
from core.config import VERSION
from ai.actions import ACTION_NAMES
from ai.config import NetworkConfig
from ai.network import SnakeNetwork
from ai.observations import FEATURE_NAMES
from services.userdata import user_data_dir


class ModelStore:
    def __init__(self, root: Path | None = None):
        self.root = root if root is not None else user_data_dir() / "models"

    def path(self, name):
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name) is None:
            raise ValueError("Model name: 1–40 letters, digits, underscores or hyphens")
        target = self.root / name
        if target.resolve().parent != self.root.resolve() or target.is_symlink():
            raise ValueError("Model must remain inside the models directory")
        return target

    def names(self):
        if not self.root.exists():
            return []
        return sorted(path.name for path in self.root.iterdir()
                      if path.is_dir() and not path.is_symlink() and (path / "network.json").is_file()
                      and re.fullmatch(r"[A-Za-z0-9_-]{1,40}", path.name))

    def save(self, name, network, *, overwrite=False, demo_steps=0, reinforcement_learning=False):
        folder = self.path(name)
        if folder.exists() and not overwrite:
            raise FileExistsError("Model already exists; confirm replacement before saving")
        folder.mkdir(parents=True, exist_ok=True)
        old_metadata = folder / "network.json"
        old_file = None
        if old_metadata.is_file():
            try:
                old_file = json.loads(old_metadata.read_text(encoding="utf-8")).get("weights_file")
            except (ValueError, OSError):
                pass
        token = uuid4().hex
        weights = folder / f"weights-{token}.pt"
        temporary = folder / f"metadata-{token}.tmp"
        committed = False
        try:
            torch.save(network.state_dict(), weights)
            metadata = dict(format_version=1, application_version=VERSION,
                created_utc=datetime.now(timezone.utc).isoformat(), architecture=network.config.to_dict(),
                features=list(FEATURE_NAMES), actions=list(ACTION_NAMES), weights_file=weights.name,
                sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                manual_edits=[list(item) for item in sorted(network.edited_parameters,
                             key=lambda item: (item[0], item[1], -1 if item[2] is None else item[2]))],
                supervised_demo_steps=demo_steps, reinforcement_learning=reinforcement_learning)
            temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(old_metadata)  # Atomic commit; weights already exist.
            committed = True
            if isinstance(old_file, str) and re.fullmatch(r"weights-[0-9a-f]{32}\.pt", old_file):
                old_path = folder / old_file
                if old_path.resolve().parent == folder.resolve():
                    old_path.unlink(missing_ok=True)
            return folder
        finally:
            temporary.unlink(missing_ok=True)
            if not committed:
                weights.unlink(missing_ok=True)

    def load(self, name):
        """Load only user-trusted local files; never fall back to unsafe pickle."""
        folder = self.path(name)
        metadata_path = folder / "network.json"
        try:
            if metadata_path.stat().st_size > 64_000:
                raise ValueError("Model metadata is too large")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (not isinstance(metadata, dict) or metadata.get("format_version") != 1
                    or metadata.get("features") != list(FEATURE_NAMES)
                    or metadata.get("actions") != list(ACTION_NAMES)):
                raise ValueError("Incompatible observation/action model")
            config = NetworkConfig.from_dict(metadata.get("architecture"))
            demo_steps = metadata.get("supervised_demo_steps", 0)
            if type(demo_steps) is not int or demo_steps < 0:
                raise ValueError("Invalid supervised update count")
            filename = metadata.get("weights_file")
            if not isinstance(filename, str) or not re.fullmatch(r"weights-[0-9a-f]{32}\.pt", filename):
                raise ValueError("Invalid weights filename")
            weights = folder / filename
            if weights.is_symlink() or not weights.resolve().is_relative_to(folder.resolve()):
                raise ValueError("Weights must be inside the model directory")
            if weights.stat().st_size > 8_000_000:
                raise ValueError("Weights file is too large")
            if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata.get("sha256"):
                raise ValueError("Weights checksum mismatch")
            with zipfile.ZipFile(weights) as archive:
                if sum(item.file_size for item in archive.infolist()) > 16_000_000:
                    raise ValueError("Expanded weights archive is too large")
            state = torch.load(weights, map_location="cpu", weights_only=True)
            model = SnakeNetwork(config)
            expected = model.state_dict()
            if not isinstance(state, dict) or set(state) != set(expected):
                raise ValueError("Weights do not match the architecture")
            for key, value in state.items():
                if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                        or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                    raise ValueError("Invalid parameter tensor: " + key)
            model.load_state_dict(state, strict=True)
            for item in metadata.get("manual_edits", []):
                if not isinstance(item, list) or len(item) != 3:
                    raise ValueError("Invalid manual-edit metadata")
                layer, neuron, incoming = item
                if (type(layer) is not int or not 0 <= layer < len(model.layers)
                        or type(neuron) is not int or not 0 <= neuron < model.layers[layer].out_features
                        or incoming is not None and (type(incoming) is not int or not 0 <= incoming < model.layers[layer].in_features)):
                    raise ValueError("Manual-edit index out of range")
                model.edited_parameters.add(tuple(item))
            return model, metadata
        except (OSError, RuntimeError, EOFError, TypeError, KeyError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
            raise ValueError("Cannot load model: " + str(error)) from error
