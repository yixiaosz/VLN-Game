"""Offline trace checks: no simulator, GUI, credentials, or API calls required.

Run with the vlngame environment: python -m unittest discover -s tests -v
"""
import base64
from concurrent.futures import Future
from contextlib import ExitStack, redirect_stdout
import hashlib
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

# chat_utils parses argv on import in the reference implementation.
with patch.object(sys, 'argv', [sys.argv[0]]):
    import utils.equ_ranking as ranking
    import utils.chat_utils as chat
from utils.equ_trace import EquilibriumTraceRun, json_value


class ImmediateExecutor:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


def jpeg(value=80, size=(960, 480)):
    buffer = BytesIO()
    Image.new('RGB', size, (value, 70, 120)).save(buffer, format='JPEG')
    return buffer.getvalue()


class Replies:
    def __init__(self, generator, discriminator, encoded):
        self.generator = iter(generator)
        self.discriminator = {k: iter(v) for k, v in discriminator.items()}
        self.image_to_candidate = {value: key for key, value in encoded.items()}
        self.calls = []

    def __call__(self, history, json_mode=True):
        images = [item['image_url']['url'].split(',', 1)[1]
                  for item in history[-1]['content'] if item['type'] == 'image_url']
        self.calls.append((json_mode, images))
        if json_mode:
            result = next(self.generator)
            if isinstance(result, int):
                result = json.dumps({'object_id': str(result)})
        else:
            key = self.image_to_candidate[images[0]] if len(images) == 1 else -1
            result = next(self.discriminator[key])
        if isinstance(result, Exception):
            raise result
        return result


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='equ-trace-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logger = logging.getLogger('equ-trace-offline-tests')
        self.logger.addHandler(logging.NullHandler())

    def run_search(self, ids, generator, discriminator, traced=True, sizes=None):
        data = {cid: jpeg(60 + n * 30, (sizes or {}).get(cid, (960, 480)))
                for n, cid in enumerate(ids)}
        encoded = {cid: base64.b64encode(value).decode('utf-8') for cid, value in data.items()}
        replies = Replies(generator, discriminator, encoded)
        trace_run = EquilibriumTraceRun(self.root, {'mode': 'vlm_game'},
                                       {'generator': 'G', 'discriminator': 'D'}, self.logger) if traced else None
        episode = trace_run.start_episode(1, '17', 'scene.glb', 'cabinet') if traced else None
        search = episode.start_search(80, ids) if traced else None
        if traced:
            for position, cid in enumerate(ids):
                source = self.root / f'Vis-{cid}.jpeg'
                source.write_bytes(data[cid])
                search.archive_image(cid, position, source, data[cid])
                # Reproduce the shared-file overwrite after the buffer was read.
                source.write_bytes(jpeg(220, (640, 800)))
        snapshots = []
        error = None
        result = None
        with ExitStack() as stack:
            stack.enter_context(patch.object(ranking, 'chat_with_gpt', return_value='{"target":"cabinet"}'))
            stack.enter_context(patch.object(ranking, 'chat_with_gpt4v', replies))
            stack.enter_context(patch.object(ranking.concurrent.futures, 'ThreadPoolExecutor', ImmediateExecutor))
            # Reverse completion order deliberately; do not confuse submission
            # sample IDs with their completion order in the trace.
            stack.enter_context(patch.object(ranking.concurrent.futures, 'as_completed', lambda fs: iter(reversed(list(fs)))))
            stack.enter_context(redirect_stdout(StringIO()))
            obj = ranking.Equilibrium_Ranking('cabinet', self.logger, episode_trace=episode)
            original_update = obj._update_policy

            def capture(Q, initial, lambd, eta, t):
                policy = original_update(Q, initial, lambd, eta, t)
                if t in (125, 250, 375, 500):
                    snapshots.append({k: v.copy() for k, v in policy.items()})
                return policy

            obj._update_policy = capture
            with np.errstate(invalid='ignore'):
                try:
                    result = obj.equilibrium_search([encoded[cid] for cid in ids], ids, trace=search)
                except Exception as exc:
                    error = type(exc)
        if episode:
            episode.finish('error' if error else 'completed', 140,
                           metrics={'success': 0.0, 'spl': 0.0, 'distance_to_goal': 2.0})
            # Reject non-standard JSON numbers, rather than accepting bare NaN.
            events = [json.loads(line, parse_constant=lambda s: self.fail(s))
                      for line in episode.path.read_text().splitlines()]
        else:
            events = []
        return result, error, snapshots, replies.calls, events, trace_run

    def compare(self, ids, generator, discriminator, **kwargs):
        untraced = self.run_search(ids, generator, discriminator, traced=False, **kwargs)
        traced = self.run_search(ids, generator, discriminator, **kwargs)
        self.assertEqual(untraced[:2], traced[:2])
        self.assertEqual(untraced[3], traced[3], 'Requests or image bytes changed')
        self.assertEqual(len(untraced[2]), len(traced[2]))
        for left, right in zip(untraced[2], traced[2]):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                np.testing.assert_equal(left[key], right[key])
        return traced

    def test_single_candidate_support_checkpoints_and_exact_images(self):
        result, error, _, calls, events, run = self.compare(
            [15, 51], [15, 15, -1, 15, 15],
            {15: ['correct'] * 4 + ['incorrect']}, sizes={51: (640, 800)})
        self.assertEqual(result, 15)
        self.assertIsNone(error)
        raw = [e for e in events if e['event'] == 'vlm_response']
        self.assertEqual(len(raw), 10)
        self.assertEqual(len({e['sample_id'] for e in raw}), 10)
        self.assertEqual(raw[0]['sample_id'], 'g-5')
        self.assertEqual(raw[0]['completion_order'], 1)
        checkpoints = [e for e in events if e['event'] == 'policy_snapshot']
        self.assertEqual([e['update_iteration'] for e in checkpoints], [0, 125, 250, 375, 500])
        np.testing.assert_allclose(checkpoints[-1]['pi_G']['correct'], [0.9999999426009226, 5.7399077383839185e-8])
        layout = next(e for e in events if e['event'] == 'policy_layout')
        self.assertEqual(layout['policy_candidate_order'], [15, -1])
        self.assertEqual(layout['discriminator_key_to_candidate_id'], {'0': 15, '-1': -1})
        aggregate = next(e for e in events if e['event'] == 'discriminator_aggregate')
        self.assertEqual(aggregate['reject_all']['source'], 'derived_complement')
        images = [e for e in events if e['event'] == 'image_input']
        self.assertFalse(images[1]['expected_dimensions'])
        for image, encoded in zip(images, calls[0][1]):
            archived = (run.root / image['archive_path']).read_bytes()
            self.assertEqual(archived, base64.b64decode(encoded))
            self.assertEqual(hashlib.sha256(archived).hexdigest(), image['sha256'])

    def test_multicandidate_accumulation_and_nans_are_preserved(self):
        result, error, _, calls, events, _ = self.compare(
            [15, 51], [15, 15, 15, 51, 51],
            {15: ['correct'] * 4 + ['incorrect'],
             51: ['correct'] * 2 + ['incorrect'] * 3,
             -1: ['incorrect'] * 5})
        self.assertIsNone(error)
        self.assertEqual(len(calls), 20)
        self.assertEqual(len([e for e in events if e['event'] == 'vlm_response']), 20)
        aggregate = next(e for e in events if e['event'] == 'discriminator_aggregate')
        self.assertEqual(aggregate['counts_used']['1'], 6)
        self.assertEqual(aggregate['denominator_used'], 5)
        self.assertEqual(aggregate['p_vlm_before_bias']['1'], [1.2, 1 - 1.2])
        self.assertTrue(aggregate['values_outside_probability_range'])
        batches = aggregate['candidate_batches']
        self.assertEqual([len(b['new_sample_ids']) for b in batches], [5, 5])
        self.assertEqual([len(b['sample_pool_ids_used_by_current_code']) for b in batches], [5, 10])
        checkpoints = [e for e in events if e['event'] == 'policy_snapshot']
        self.assertLess(checkpoints[0]['pi_D']['1'][1], 0)
        self.assertFalse(checkpoints[0]['diagnostics']['pi_D']['1']['nonnegative'])
        self.assertIn('NaN', json.dumps(checkpoints[-1]))
        end = next(e for e in events if e['event'] == 'search_end')
        self.assertEqual(end['selected_candidate_id'], result)

    def test_three_candidates_and_full_set_reject_query(self):
        _, _, _, calls, events, _ = self.compare(
            [15, 51, 72], [15, 15, 51, 72, -1],
            {15: ['correct'] + ['incorrect'] * 4,
             51: ['correct'] + ['incorrect'] * 4,
             72: ['incorrect'] * 5, -1: ['incorrect'] * 5})
        self.assertEqual(len(calls), 25)
        aggregate = next(e for e in events if e['event'] == 'discriminator_aggregate')
        self.assertEqual([b['correct_count_used'] for b in aggregate['candidate_batches']], [1, 2, 2])
        self.assertEqual([len(b['sample_pool_ids_used_by_current_code']) for b in aggregate['candidate_batches']], [5, 10, 15])
        self.assertEqual(aggregate['reject_all']['source'], 'queried_all_input_images')
        self.assertEqual(len(calls[-1][1]), 3)

    def test_all_reject_skips_policies(self):
        result, _, _, calls, events, _ = self.compare([15], [-1] * 5, {})
        self.assertEqual(result, -1)
        self.assertEqual(len(calls), 5)
        self.assertFalse(any(e['event'] == 'policy_snapshot' for e in events))
        end = next(e for e in events if e['event'] == 'search_end')
        self.assertEqual(end['status'], 'early_reject')
        self.assertEqual(end['equilibrium_updates_completed'], 0)

    def test_failed_samples_keep_reference_denominators(self):
        _, _, _, _, events, _ = self.compare(
            [15], [15, 15, RuntimeError('mock failure'), 15, -1],
            {15: ['correct', RuntimeError('mock failure'), 'incorrect', 'correct', RuntimeError('mock failure')]})
        gen = next(e for e in events if e['event'] == 'generator_aggregate')
        disc = next(e for e in events if e['event'] == 'discriminator_aggregate')
        self.assertEqual(gen['denominator_used'], 4)
        self.assertEqual(disc['denominator_used'], 5)
        self.assertEqual(disc['counts_used']['0'], 2)
        errors = [e for e in events if e['event'] == 'vlm_response' and e['status'] == 'error']
        self.assertEqual(len(errors), 3)

    def test_invalid_generator_response_is_archived_and_raises(self):
        _, error, _, _, events, _ = self.compare([15], ['not JSON'] * 5, {})
        self.assertEqual(error, ValueError)
        self.assertEqual(len([e for e in events if e['event'] == 'vlm_response']), 5)
        self.assertTrue(any(e['event'] == 'sample_parse_error' for e in events))
        self.assertEqual(next(e for e in events if e['event'] == 'search_end')['status'], 'error')

    def test_invalid_discriminator_word_is_not_reinterpreted(self):
        _, _, _, _, events, _ = self.compare([15], [15] * 5, {15: ['maybe'] * 5})
        aggregate = next(e for e in events if e['event'] == 'discriminator_aggregate')
        self.assertEqual(aggregate['counts_used']['0'], 0)
        parsed = [e for e in events if e['event'] == 'sample_parsed' and e['role'] == 'discriminator']
        self.assertTrue(all(not e['recognized_verdict'] for e in parsed))

    def test_unique_runs_multiple_searches_and_flushed_partial_episode(self):
        first = EquilibriumTraceRun(self.root, {}, {}, self.logger)
        second = EquilibriumTraceRun(self.root, {}, {}, self.logger)
        self.assertNotEqual(first.root, second.root)
        episode = first.start_episode(1, '0', 'scene', 'instruction')
        one = episode.start_search(20, [15])
        one.archive_image(15, 0, 'Vis-15.jpeg', jpeg(40))
        one.finish('early_reject', selected_candidate_id=-1)
        two = episode.start_search(80, [15])
        two.archive_image(15, 0, 'Vis-15.jpeg', jpeg(190))
        events = [json.loads(line) for line in episode.path.read_text().splitlines()]
        images = [e for e in events if e['event'] == 'image_input']
        self.assertNotEqual(images[0]['archive_path'], images[1]['archive_path'])
        self.assertNotEqual(images[0]['sha256'], images[1]['sha256'])
        self.assertFalse(any(e['event'] == 'search_end' and e['search_id'] == 2 for e in events))
        self.assertFalse(any(e['event'] == 'episode_end' for e in events))
        episode.finish('interrupted', 80)

    def test_nonfinite_numpy_serialization_is_explicit(self):
        result = json.loads(json.dumps(json_value({np.int64(2): np.array([np.nan, np.inf, -np.inf])}), allow_nan=False))
        self.assertEqual(result, {'2': ['NaN', 'Infinity', '-Infinity']})

    def test_manifest_configuration_excludes_credentials(self):
        with patch.dict(os.environ, {'VLN_MODEL': 'example-model', 'VLN_MODEL_OPTIONS':
                                     '{"temperature":0.7,"api_key":"do-not-store","thinking":{"type":"disabled"}}'}), \
                patch.object(chat, 'api_base', 'https://user:password@example.com/v1?token=secret'):
            config = chat.trace_configuration()
        run = EquilibriumTraceRun(self.root, {'api_configuration': config}, {}, self.logger)
        text = (run.root / 'run.json').read_text()
        self.assertNotIn('do-not-store', text)
        self.assertNotIn('password', text)
        self.assertNotIn('token=secret', text)
        self.assertEqual(config['requests']['generator']['model'], 'example-model')
        self.assertEqual(config['requests']['generator']['temperature'], 0.7)


if __name__ == '__main__':
    unittest.main()
