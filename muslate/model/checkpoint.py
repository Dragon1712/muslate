"""Immutable, atomically published research model checkpoints."""
from dataclasses import asdict
from pathlib import Path
import os
import tempfile
import torch
from muslate.config import WorldModelConfig
from muslate.model.world_model import WorldModel


def save_checkpoint(model: WorldModel, path: str | Path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    payload = dict(format_version=1, config=asdict(model.cfg), n_items=model.n_items,
                   slate_size=model.slate_size, true_state_dim=model.true_state_dim,
                   state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                   training_history=getattr(model, 'training_history', []), metadata=metadata)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.checkpoint-')
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication; existing files are never replaced.
    finally:
        os.unlink(temporary)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[WorldModel, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload['format_version'] != 1:
        raise ValueError('unsupported checkpoint format')
    model = WorldModel(WorldModelConfig(**payload['config']), payload['n_items'],
                       payload['slate_size'], payload['true_state_dim']).to(device)
    model.load_state_dict(payload['state_dict'])
    model.training_history = payload['training_history']
    return model.eval(), payload['metadata']
