"""Exact-instruction, versioned parsing for Jev only (no model imports)."""
import gzip
import hashlib
import json
import os
from pathlib import Path

DEFAULT_TABLE = Path(__file__).resolve().parents[1] / 'configs/jev_instruction_parses.v1.json'
DEFAULT_DATASET = Path(__file__).resolve().parents[1] / 'data/datasets/vlobjectnav_hm3d_v4/val/val.json.gz'
SCHEMA = 'jev_exact_instruction_parses_v1'


def is_jev():
    return os.environ.get('VLN_MODEL') == 'typesafe/jev-1.14'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key in instruction table')
        result[key] = value
    return result


def valid_parse(value):
    return (isinstance(value, dict) and isinstance(value.get('target'), str)
            and isinstance(value.get('landmark', []), list)
            and all(isinstance(item, str) for item in value.get('landmark', [])))


def dataset_instructions(path=DEFAULT_DATASET):
    """Read only instructions for model input; hash source files for provenance."""
    path = Path(path)
    files = [path] + sorted((path.parent / 'content').glob('*.json.gz'))
    instructions, fingerprints, episodes = [], {}, 0
    for source in files:
        raw = source.read_bytes()
        fingerprints[str(source.relative_to(path.parent))] = sha256(raw)
        data = json.loads(gzip.decompress(raw))
        for episode in data['episodes']:
            text = episode['instruction_text']
            if not isinstance(text, str):
                raise ValueError('Dataset instruction must be a string')
            instructions.append(text)
            episodes += 1
    texts = sorted(set(instructions))
    return texts, {'episode_count': episodes, 'unique_instructions': len(texts),
                   'files_sha256': fingerprints, 'instructions_sha256': sha256(canonical(texts))}


class InstructionLookup:
    def __init__(self, path=None, dataset_path=DEFAULT_DATASET):
        self.path = Path(path or os.environ.get('VLN_JEV_PARSE_TABLE') or DEFAULT_TABLE)
        raw = self.path.read_bytes()
        expected = self.path.with_suffix(self.path.suffix + '.sha256').read_text().strip()
        self.hash = sha256(raw)
        if self.hash != expected:
            raise ValueError('Frozen Jev instruction table checksum mismatch')
        data = json.loads(raw, object_pairs_hook=unique_keys)
        if data.get('schema') != SCHEMA or data.get('status') != 'frozen':
            raise ValueError('Jev requires a frozen, supported instruction table')
        texts, inventory = dataset_instructions(dataset_path)
        self.entries = data['entries']
        if set(self.entries) != set(texts) or data['dataset'] != inventory:
            raise ValueError('Frozen Jev table does not exactly cover the configured dataset')
        if not all(valid_parse(value) and set(value) == {'target', 'landmark'} for value in self.entries.values()):
            raise ValueError('Invalid target/landmark entry in frozen Jev table')
        self.metadata = {'protocol': SCHEMA, 'path': str(self.path), 'sha256': self.hash,
                         'entries': len(self.entries), 'dataset': inventory,
                         'parser': data['parser'], 'semantic_accuracy': 'not established by structural validation'}

    def get(self, instruction):
        if instruction not in self.entries:
            raise ValueError('Exact instruction missing from frozen Jev table; no live fallback')
        # Return a copy so an episode cannot mutate the frozen lookup in memory.
        return json.loads(json.dumps(self.entries[instruction]))
