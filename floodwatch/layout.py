"""Resolve pre-reorganization paths without replacing historical hash baselines.

Only recorded import/path/integrity-adapter edits are reversed for source checks.
Data and model artifacts are always hashed byte-for-byte, without normalization.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAYOUT = json.loads((Path(__file__).with_name('relocation.json')).read_text())


def resolve_legacy(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if path.parent == ROOT and path.suffix == '.py':
        module = LAYOUT['modules'].get(path.stem)
        if module:
            return ROOT / Path(*module.split('.')).with_suffix('.py')
    return path


def original_source(text, name=None):
    for old, new in reversed(LAYOUT.get('source_edits', {}).get(name, [])):
        text = text.replace(new, old)
    for old, new in reversed(LAYOUT['replacements']):
        text = text.replace(new, old)
    return text


def legacy_bytes(path):
    resolved = resolve_legacy(path)
    if resolved.suffix == '.py' and (resolved.parent == ROOT or ROOT/'floodwatch' in resolved.parents):
        return original_source(resolved.read_text(), resolved.name).encode()
    return resolved.read_bytes()


def legacy_digest(path):
    return hashlib.sha256(legacy_bytes(path)).hexdigest()
