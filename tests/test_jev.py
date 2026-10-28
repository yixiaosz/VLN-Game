"""Jev full-distribution, exact-lookup, transport and legacy-solver regressions."""
import base64
from contextlib import redirect_stdout
import gzip
from io import StringIO
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import test_chat_utils as http
import test_equ_trace as fixtures
from utils import jev_client as client
from utils import jev_ranking as ranking
from utils import instruction_lookup as lookup
from utils.equ_ranking import Equilibrium_Ranking
from utils.equ_trace import EquilibriumTraceRun
from utils.equ_report import episode_report


def envelope(probabilities=None, noul=None):
    answer = ({'type': 'choice', 'choice': max(probabilities, key=probabilities.get),
               'probabilities': probabilities, 'confidence': 0.0} if probabilities is not None
              else {'type': 'noul', 'noul': noul})
    return {'model': 'typesafe/jev-1.14-20261014', 'provider': 'TypeSafe', 'answers': {'decision': answer}}


class SolverTests(unittest.TestCase):
    def test_array_solver_matches_existing_dictionary_updates_and_checkpoints(self):
        rng = np.random.RandomState(13)
        for K in (2, 4, 8):
            g, d = rng.dirichlet(np.ones(K)), rng.uniform(0, 1, K)
            if K == 4:
                g = np.array([0, 0, 0, 1.])
                d = np.array([0, 1, .5, .7])
            seen = {}
            actualG, actualD = ranking.solve_equilibrium(g, d, lambda t, G, D, QG, QD: seen.update({t: (G.copy(), D.copy(), QG.copy(), QD.copy())}))
            obj = object.__new__(Equilibrium_Ranking)
            G0 = obj._normalize_initial_policy({'correct': list(g+.05), 'incorrect': list(1-g+.05)})
            D0 = obj._normalize_initial_policy({i: [v+.05, 1-v+.05] for i, v in enumerate(d)})
            G, D = G0, D0
            QG = {v: np.zeros(K) for v in G}
            QD = {i: np.zeros(2) for i in D}
            for t in range(1, 501):
                for vi, v in enumerate(G):
                    for i in range(K):
                        QG[v][i] += D[i][vi]/(2*t)
                for i in range(K):
                    for vi, v in enumerate(G):
                        QD[i][vi] += G[v][i]/(2*t)
                G = obj._update_policy(QG, G0, .1, .1, t)
                D = obj._update_policy(QD, D0, .1, .1, t)
                if t in seen:
                    for expected, actual in zip((np.array(list(G.values())), np.array(list(D.values())),
                                                 np.array(list(QG.values())), np.array(list(QD.values()))), seen[t]):
                        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
            self.assertEqual(actualG.shape, (2, K))
            self.assertEqual(actualD.shape, (K, 2))
            np.testing.assert_allclose(actualG.sum(axis=1), 1)
            np.testing.assert_allclose(actualD.sum(axis=1), 1)
            self.assertEqual(set(seen), {0, 125, 250, 375, 500})

    def test_malformed_vectors_rejected(self):
        for g, d in (([.5, .5], [1]), ([1, 1], [.5, .5]), ([np.nan, 0], [0, 1]), ([1, 0], [-1, 1])):
            with self.assertRaises(ValueError):
                ranking.solve_equilibrium(g, d)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.payload = client.build_request('exact instruction', [42, 17], ['YQ==', 'Yg=='], 'choice')
        self.patch = patch.dict(os.environ, {'VLN_API_KEY': 'unit-jev-key', 'VLN_MODEL_OPTIONS': '{"temperature":9,"thinking":true}'})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_native_body_reused_and_missing_option_retries_inside_one_budget(self):
        bad, good = envelope({'42': .8, '-1': .2}), envelope({'42': .2, '17': .6, '-1': .2})
        events = []
        with patch.object(client.requests, 'post', side_effect=[http.response(body=json.dumps(bad)), http.response(body=json.dumps(good))]) as post, \
                patch.object(http.chat.time, 'sleep'), redirect_stdout(StringIO()):
            data, parsed = client.request_decision(self.payload, 'generator', lambda event, **kw: events.append((event, kw)))
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].kwargs['data'], post.call_args_list[1].kwargs['data'])
        body = json.loads(post.call_args.kwargs['data'])
        self.assertEqual(set(body), {'model', 'state', 'questions'})
        self.assertNotIn('temperature', body)
        self.assertEqual(post.call_args.kwargs['timeout'], (10., 90.))
        self.assertEqual(parsed['probabilities'], good['answers']['decision']['probabilities'])
        self.assertEqual(sum(e == 'jev_raw_response' for e, _ in events), 2)
        self.assertIn('jev_response_structure', [kw.get('error_category') for _, kw in events])

    def test_invalid_probabilities_and_mass_exhaust_without_zero_fill(self):
        for bad in ({'42': .2, '17': .2, '-1': .2}, {'42': True, '17': 0, '-1': 0},
                    {'42': float('nan'), '17': .5, '-1': .5}, {'42': -1, '17': 1, '-1': 1}):
            # Set a valid textual choice; validation must reject the distribution itself.
            data = envelope({'42': .5, '17': .4, '-1': .1})
            data['answers']['decision']['probabilities'] = bad
            with patch.object(client.requests, 'post', side_effect=lambda *a, **k: http.response(body=json.dumps(data))) as post, \
                    patch.object(http.chat.time, 'sleep'), redirect_stdout(StringIO()), self.assertRaises(http.chat.APIRequestError):
                client.request_decision(self.payload, 'generator')
            self.assertEqual(post.call_count, 3)

    def test_noul_scope_and_rejection_polarity(self):
        payload = client.build_request('full description', [42], ['YQ=='], 'match')
        self.assertEqual(len([p for p in payload['state'] if isinstance(p, dict)]), 1)
        data = envelope(noul=.8)
        self.assertEqual(client.validate_response(data, payload), {'noul': .8})
        reject = client.build_request('full description', [42, 17], ['YQ==', 'Yg=='], 'reject')
        self.assertIn('none', reject['questions']['decision']['instructions'])
        for invalid in (-.1, 1.1, True, None):
            with self.assertRaises(ValueError):
                client.validate_response(envelope(noul=invalid), payload)

    def test_wrong_model_and_unoffered_choice_rejected(self):
        data = envelope({'42': .5, '17': .4, '-1': .1})
        data['model'] = 'deepseek-flash'
        with self.assertRaises(ValueError):
            client.validate_response(data, self.payload)
        data['model'] = client.MODEL
        data['answers']['decision']['choice'] = '4'
        with self.assertRaises(ValueError):
            client.validate_response(data, self.payload)

    def test_non_json_response_retries_and_permanent_error_stops(self):
        good = envelope({'42': .5, '17': .4, '-1': .1})
        with patch.object(client.requests, 'post', side_effect=[http.response(body='<html>bad gateway</html>'),
                    http.response(body=json.dumps(good))]) as post, patch.object(http.chat.time, 'sleep'), redirect_stdout(StringIO()):
            client.request_decision(self.payload, 'generator')
        self.assertEqual(post.call_count, 2)
        with patch.object(client.requests, 'post', return_value=http.response(status=403, body='Forbidden')) as post, \
                redirect_stdout(StringIO()), self.assertRaises(http.chat.APIRequestError):
            client.request_decision(self.payload, 'generator')
        self.assertEqual(post.call_count, 1)


class LookupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root/'val.json.gz'
        self.dataset.write_bytes(gzip.compress(json.dumps({'episodes': [{'instruction_text': "Exact 'instruction'."}]}).encode()))
        texts, inventory = lookup.dataset_instructions(self.dataset)
        self.table = {'schema': lookup.SCHEMA, 'status': 'frozen', 'dataset': inventory,
                      'parser': {'model': 'deepseek-flash'}, 'entries': {texts[0]: {'target': "object's door", 'landmark': []}}}
        self.path = self.root/'table.json'

    def publish(self):
        raw = lookup.canonical(self.table)
        self.path.write_bytes(raw)
        self.path.with_suffix('.json.sha256').write_text(lookup.sha256(raw))

    def test_exact_keys_no_live_fallback_and_no_mutable_shared_entry(self):
        self.publish()
        table = lookup.InstructionLookup(self.path, self.dataset)
        value = table.get("Exact 'instruction'.")
        value['landmark'].append('mutation')
        self.assertEqual(table.get("Exact 'instruction'.")['landmark'], [])
        with self.assertRaises(ValueError):
            table.get("exact 'instruction'.")
        obj = ranking.JevEquilibriumRanking("Exact 'instruction'.", lookup=table)
        self.assertEqual(__import__('ast').literal_eval(obj.response_message)['target'], "object's door")

    def test_checksum_coverage_and_dataset_identity(self):
        self.publish()
        self.path.write_bytes(self.path.read_bytes()+b' ')
        with self.assertRaises(ValueError):
            lookup.InstructionLookup(self.path, self.dataset)
        self.table['entries'] = {}
        self.publish()
        with self.assertRaises(ValueError):
            lookup.InstructionLookup(self.path, self.dataset)
        with self.assertRaises(ValueError):
            json.loads('{"same":1,"same":2}', object_pairs_hook=lookup.unique_keys)


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.obj = object.__new__(ranking.JevEquilibriumRanking)
        self.obj.text_description = 'unchanged full description'
        self.images = [base64.b64encode(fixtures.jpeg(70)).decode(), base64.b64encode(fixtures.jpeg(90)).decode()]

    def run_search(self, samples=5, game=False, fail=False, single=False):
        run = EquilibriumTraceRun(self.tmp.name, {'mode': 'vlm_game' if game else 'vlm_rank', 'backend': 'jev',
            'discriminator_aggregation': 'candidate_local_noul_mean_v1'}, {}, logging)
        ep = run.start_episode(1, '0', 'synthetic', self.obj.text_description)
        ids = [42] if single else [42, 17]
        trace = ep.start_search(20, ids)
        for pos, cid in enumerate(ids):
            trace.archive_image(cid, pos, 'synthetic', base64.b64decode(self.images[pos]))
        counts = {'choice': 0, 'match': 0, 'reject': 0}
        def answer(payload, role, observer=None, cancel_event=None):
            q = payload['questions']['decision']
            if q['type'] == 'choice':
                n = counts['choice']
                counts['choice'] += 1
                if fail and n == 2:
                    raise http.chat.APIRequestError('required sample failed')
                probs = ({'42': .9, '-1': .1} if single else
                         {'42': .49 if n < 3 else .99, '17': .51 if n < 3 else .01, '-1': 0.})
                return envelope(probs), {'probabilities': probs}
            is_reject = q['instructions'] == client.REJECT_INSTRUCTION
            counts['reject' if is_reject else 'match'] += 1
            parts = [p for p in payload['state'] if isinstance(p, dict)]
            self.assertEqual(len(parts), len(ids) if is_reject else 1)
            return envelope(noul=.2 if is_reject else .8), {'noul': .2 if is_reject else .8}
        with patch.object(ranking, 'request_decision', side_effect=answer), \
                patch.object(ranking.concurrent.futures, 'ThreadPoolExecutor', lambda **kw: fixtures.ImmediateExecutor()), \
                patch.object(ranking.concurrent.futures, 'as_completed', lambda fs: iter(reversed(list(fs)))):
            if fail:
                with self.assertRaises(http.chat.APIRequestError):
                    self.obj.generator_search(self.images, ids, samples, trace)
                selected = None
            else:
                selected = (self.obj.equilibrium_search(self.images[:len(ids)], ids, trace) if game else
                            self.obj.generator_search(self.images, ids, samples, trace))
        ep.finish('error' if fail else 'completed', 20)
        events = [json.loads(line) for line in ep.path.read_text().splitlines()]
        return selected, counts, events, episode_report(ep.path)

    def test_probability_mean_is_not_plurality_and_report_does_not_call_it_votes(self):
        selected, counts, events, report = self.run_search()
        self.assertEqual(selected, 42)  # Three hard choices favor 17, but pooled probabilities favor 42.
        self.assertEqual(counts, {'choice': 5, 'match': 0, 'reject': 0})
        aggregate = next(e for e in events if e['event'] == 'generator_aggregate')
        self.assertEqual(aggregate['policy_candidate_order'], [42, 17, -1])
        self.assertAlmostEqual(aggregate['p_vlm_correct'][0], .69)
        self.assertIn('NATIVE PROBABILITIES', report)
        self.assertNotIn('Votes', report)

    def test_single_query_generator_uses_one_distribution(self):
        selected, counts, events, _ = self.run_search(samples=1)
        self.assertEqual(selected, 17)
        self.assertEqual(counts, {'choice': 1, 'match': 0, 'reject': 0})
        self.assertFalse(any(e['event'] == 'policy_snapshot' for e in events))

    def test_game_full_support_arrays_and_single_candidate_complement(self):
        for single in (False, True):
            selected, counts, events, report = self.run_search(game=True, single=single)
            self.assertEqual(counts, {'choice': 5, 'match': 5 if single else 10, 'reject': 0 if single else 5})
            snapshots = [e for e in events if e['event'] == 'policy_snapshot']
            self.assertEqual([e['update_iteration'] for e in snapshots], [0, 125, 250, 375, 500])
            self.assertEqual(np.asarray(snapshots[0]['pi_G']).shape, (2, 2 if single else 3))
            self.assertEqual(np.asarray(snapshots[0]['pi_D']).shape, (2 if single else 3, 2))
            self.assertEqual(next(e for e in events if e['event'] == 'search_end')['equilibrium_updates_completed'], 500)
            self.assertIn('ARRAY EQUILIBRIUM', report)
            self.assertNotIn('POLICY ANOMALY', report)

    def test_required_failure_has_no_partial_aggregate(self):
        _, _, events, _ = self.run_search(fail=True)
        self.assertFalse(any(e['event'] == 'generator_aggregate' for e in events))
        self.assertEqual(next(e for e in events if e['event'] == 'search_end')['status'], 'error')


class LauncherTests(unittest.TestCase):
    def test_custom_registry_preserved_and_model_switch_resets_request_settings(self):
        import demo
        deepseek = next(m for m in demo.DEFAULT_MODELS if m['id'] == 'deepseek-flash')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'VLN_MODELS': json.dumps([deepseek]), 'GSA_PATH': directory,
                'OPENROUTER_API_KEY': 'unit-router', 'DEEPSEEK_API_KEY': 'unit-deepseek',
                'VLN_MODEL_OPTIONS': '{"old_setting":true}'}):
            models = demo.load_models()
            self.assertEqual(models[0], deepseek)
            self.assertEqual(len(models), 2)
            jev = models[1]
            command, jev_env = demo.build_launch('llm', 3, jev, 0)
            self.assertEqual(jev_env['VLN_API_KEY'], 'unit-router')
            self.assertEqual(json.loads(jev_env['VLN_MODEL_OPTIONS']), {})
            self.assertEqual(jev_env['VLN_MODEL_SUPPORTS_TEMPERATURE'], '0')
            with patch.dict(os.environ, jev_env):
                _, next_env = demo.build_launch('llm', 3, deepseek, 0)
            self.assertEqual(next_env['VLN_API_KEY'], 'unit-deepseek')
            self.assertEqual(next_env['VLN_MODEL'], 'deepseek-flash')
            self.assertEqual(next_env['VLN_MODEL_SUPPORTS_TEMPERATURE'], '1')
            self.assertEqual(json.loads(next_env['VLN_MODEL_OPTIONS']), {'thinking': {'type': 'disabled'}})


if __name__ == '__main__':
    unittest.main()
