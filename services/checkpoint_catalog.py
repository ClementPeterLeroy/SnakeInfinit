"""Read-only discovery of explicitly registered Frontier checkpoint libraries."""
import json
from pathlib import Path


def checkpoint_roots(root):
    root=Path(root)
    result=[root]
    index=root.parent/'frontier_index'
    for path in sorted(index.glob('*.json')):
        try:
            if path.is_symlink() or path.stat().st_size>4000:continue
            data=json.loads(path.read_text(encoding='utf-8'))
            candidate=Path(data['checkpoint_root'])
            if data.get('schema_version')==1 and candidate.is_absolute() and candidate.is_dir() and candidate not in result:
                result.append(candidate)
        except (OSError,ValueError,KeyError,TypeError):continue
    return tuple(result)
