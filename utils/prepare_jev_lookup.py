"""Generate/resume a DeepSeek-only table, then publish a complete frozen version.

Run: python -m utils.prepare_jev_lookup --archive saved_sessions/... [--output ...]
Load private API environment first. Never put a credential in command arguments.
"""
import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from agents.system_prompt import Parsing_prompt
from utils.instruction_lookup import (
    DEFAULT_DATASET, DEFAULT_TABLE, SCHEMA, canonical, dataset_instructions, sha256, valid_parse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_TABLE)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + '.sha256').exists():
        parser.error('Refusing to overwrite a frozen table/version')
    if not os.environ.get('DEEPSEEK_API_KEY'):
        parser.error('DEEPSEEK_API_KEY is not set')
    texts, dataset = dataset_instructions(args.dataset)
    # Isolated preparation process: use precisely the existing direct DeepSeek parser settings.
    os.environ.update(VLN_API_KEY=os.environ['DEEPSEEK_API_KEY'], VLN_API_BASE='https://api.deepseek.com',
                      VLN_MODEL='deepseek-flash', VLN_MODEL_SUPPORTS_TEMPERATURE='1',
                      VLN_MODEL_OPTIONS='{"thinking":{"type":"disabled"}}')
    saved_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]  # Existing chat utility parses application arguments at import.
        from utils.chat_utils import chat_with_gpt, trace_configuration
        from utils import chat_utils
    finally:
        sys.argv = saved_argv
    settings = trace_configuration()
    provenance = {'schema': SCHEMA, 'dataset': dataset, 'prompt': Parsing_prompt,
                  'parser': {'model': 'deepseek-flash', 'api_base': settings['api_base'],
                             'request': settings['requests']['parsing'], 'transport': settings['transport']}}
    args.archive.mkdir(parents=True, exist_ok=True)
    config_path = args.archive / 'generation.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != provenance:
            raise ValueError('Resume settings/dataset differ from the original generation')
    else:
        config_path.write_bytes(canonical(provenance) + b'\n')
    progress = args.archive / 'responses.jsonl'
    entries = {}
    if progress.exists():
        for line in progress.read_text().splitlines():
            record = json.loads(line)
            if record['instruction'] in entries or not valid_parse(record['parsed']):
                raise ValueError('Invalid/duplicate completed entry in resume archive')
            entries[record['instruction']] = record['parsed']
    if not set(entries).issubset(texts):
        raise ValueError('Resume archive contains an unknown instruction')
    original_create = chat_utils.openai.ChatCompletion.create
    current = {}
    def capture_response(*call_args, **kwargs):
        response = original_create(*call_args, **kwargs)
        # SDK response contains provider data, not request authorization/configuration.
        with (args.archive / 'raw-responses.jsonl').open('a') as stream:
            stream.write(json.dumps({'instruction_sha256': current['hash'],
                                     'response': response.to_dict_recursive()}, ensure_ascii=False) + '\n')
        return response
    chat_utils.openai.ChatCompletion.create = capture_response
    with (args.archive / 'attempts.jsonl').open('a', buffering=1) as attempts, progress.open('a', buffering=1) as completed:
        for text in texts:
            if text in entries:
                continue
            current['hash'] = sha256(text.encode())
            def observer(event, **fields):
                attempts.write(json.dumps({'event': event, 'instruction_sha256': sha256(text.encode()),
                                           'timestamp_utc': datetime.now(timezone.utc).isoformat(), **fields}) + '\n')
            raw = chat_with_gpt([{'role': 'system', 'content': Parsing_prompt}, {'role': 'user', 'content': text}],
                                attempt_observer=observer)
            parsed = ast.literal_eval(raw)
            assert valid_parse(parsed)
            value = {'target': parsed['target'], 'landmark': parsed.get('landmark', [])}
            record = {'instruction': text, 'raw_content': raw, 'parsed': value,
                      'timestamp_utc': datetime.now(timezone.utc).isoformat()}
            completed.write(json.dumps(record, ensure_ascii=False) + '\n')
            completed.flush()
            os.fsync(completed.fileno())
            entries[text] = value
            print(f'Completed {len(entries)}/{len(texts)} instruction parses', flush=True)
    assert set(entries) == set(texts) and all(valid_parse(value) for value in entries.values())
    table = {'schema': SCHEMA, 'status': 'frozen', 'dataset': dataset, 'parser': provenance['parser'],
             'prompt_sha256': sha256(Parsing_prompt.encode()), 'entries': entries}
    data = canonical(table) + b'\n'
    # Exclusive publication: complete entries only, never overwrite an existing version.
    with args.output.open('xb') as stream:
        stream.write(data)
    with args.output.with_suffix(args.output.suffix + '.sha256').open('x') as stream:
        stream.write(sha256(data) + '\n')
    (args.archive / 'frozen-table.json').write_bytes(data)
    print(json.dumps({'frozen': True, 'entries': len(entries), 'sha256': sha256(data), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
