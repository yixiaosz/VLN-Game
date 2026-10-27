"""Generator/Ranking selection, request identity, rejection and trace lifecycle."""
import base64
from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
import unittest
from unittest.mock import patch

import test_equ_trace as fixtures
import test_chat_utils as http_fixtures
from agents.system_prompt import generator_prompt
from utils import equ_ranking as ranking
from utils import chat_utils as chat
from utils.equ_report import episode_report
from utils.equ_trace import EquilibriumTraceRun


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TraceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def run_search(self, ids, votes, samples=1):
        encoded = {cid: base64.b64encode(fixtures.jpeg(60+n*30)).decode()
                   for n, cid in enumerate(ids)}
        replies = fixtures.Replies(votes, {}, encoded)
        run = EquilibriumTraceRun(self.fixture.root,
            {'mode': 'vlm' if samples == 1 else 'vlm_rank'},
            {'generator': generator_prompt}, self.fixture.logger)
        episode = run.start_episode(1, 'test', 'scene', 'test description')
        search = episode.start_search(80, ids)
        for n, cid in enumerate(ids):
            search.archive_image(cid, n, 'synthetic.jpeg', base64.b64decode(encoded[cid]))

        def checked(history, **kwargs):
            self.assertEqual(history, [
                {'role': 'system', 'content': generator_prompt},
                {'role': 'user', 'content': [{'type': 'text', 'text': 'test description'}] + [
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+encoded[cid]}}
                    for cid in ids]}])
            return replies(history, **kwargs)

        error, result = None, None
        with ExitStack() as stack:
            stack.enter_context(patch.object(ranking, 'chat_with_gpt', return_value='{"target":"test"}'))
            stack.enter_context(patch.object(ranking, 'chat_with_gpt4v', side_effect=checked))
            stack.enter_context(patch.object(ranking.concurrent.futures, 'ThreadPoolExecutor', fixtures.ImmediateExecutor))
            stack.enter_context(patch.object(ranking.concurrent.futures, 'as_completed', lambda fs: iter(reversed(list(fs)))))
            stack.enter_context(redirect_stdout(StringIO()))
            obj = ranking.Equilibrium_Ranking('test description', self.fixture.logger)
            try:
                result = obj.generator_search([encoded[cid] for cid in ids], ids, samples, trace=search)
            except Exception as exc:
                error = exc
        self.assertIsNone(obj.trace)
        episode.finish('error' if error else 'completed', 80, error={'message': str(error)} if error else None)
        events = [json.loads(line) for line in episode.path.read_text().splitlines()]
        self.assertEqual(len(replies.calls), samples)
        self.assertTrue(all(json_mode for json_mode, _ in replies.calls), 'No Discriminator calls allowed')
        self.assertFalse(any(e['event'] in ('policy_snapshot', 'discriminator_aggregate', 'policy_layout') for e in events))
        self.assertEqual(next(e for e in events if e['event'] == 'search_end')['equilibrium_updates_completed'], 0)
        manifest = json.loads((run.root/'run.json').read_text())
        self.assertEqual(manifest['discriminator_aggregation'], 'not_applicable')
        return result, error, events, episode_report(episode.path)

    def test_generator_selects_actual_nonconsecutive_object_id(self):
        result, error, events, report = self.run_search([17, 42, 86], [42])
        self.assertIsNone(error)
        self.assertEqual(result, 42)
        self.assertEqual(next(e for e in events if e['event'] == 'generator_aggregate')['denominator_used'], 1)
        self.assertIn('Discriminator and equilibrium are not used', report)
        self.assertIn('Equilibrium updates: 0 (not applicable)', report)
        self.assertNotIn('No snapshots recorded', report)

    def test_ranking_selects_plurality_without_requiring_three_votes(self):
        result, error, events, _ = self.run_search([17, 42, 86], [42, 17, 42, 86, -1], 5)
        self.assertIsNone(error)
        self.assertEqual(result, 42)
        self.assertEqual(next(e for e in events if e['event'] == 'generator_aggregate')['denominator_used'], 5)

    def test_ranking_tie_uses_input_order_not_completion_order_or_lowest_id(self):
        result, error, events, _ = self.run_search([42, 17, 86], [17, 42, 17, 42, -1], 5)
        self.assertIsNone(error)
        self.assertEqual(result, 42)
        self.assertEqual(next(e for e in events if e['event'] == 'generator_selection')['tied_candidate_ids'], [42, 17])

    def test_rejection_is_returned_as_minus_one_in_both_baselines(self):
        for samples, votes in ((1, [-1]), (5, [-1, -1, 42, -1, 42])):
            with self.subTest(samples=samples):
                result, error, events, _ = self.run_search([17, 42], votes, samples)
                self.assertIsNone(error)
                self.assertEqual(result, -1)
                self.assertEqual(next(e for e in events if e['event'] == 'search_end')['reason'], 'generator_selected_reject')

    def test_candidate_wins_tie_with_reject_which_is_last_in_option_order(self):
        result, error, _, _ = self.run_search([42, 17], [42, -1, 42, -1, 17], 5)
        self.assertIsNone(error)
        self.assertEqual(result, 42)

    def test_invalid_ids_abort_without_turning_into_reject_votes(self):
        for content in ('{"object_id":"999"}', '{"object_id":42}', '{}', '[]'):
            with self.subTest(content=content):
                result, error, events, _ = self.run_search([17, 42], [content])
                self.assertIsNone(result)
                self.assertIsNotNone(error)
                self.assertFalse(any(e['event'] == 'generator_aggregate' for e in events))
                self.assertEqual(next(e for e in events if e['event'] == 'search_end')['status'], 'error')

    def test_failed_required_sample_aborts_without_partial_vote(self):
        result, error, events, _ = self.run_search([17, 42], [42, 42, RuntimeError('synthetic API failure'), 17, -1], 5)
        self.assertIsNone(result)
        self.assertIsInstance(error, ranking.APIRequestError)
        self.assertFalse(any(e['event'] == 'generator_aggregate' for e in events))

    def test_ranking_http_retry_does_not_add_a_sixth_logical_vote(self):
        # Exercise the real HTTP/retry client under the five-sample batch helper.
        obj = ranking.Equilibrium_Ranking.__new__(ranking.Equilibrium_Ranking)
        obj.text_description = 'test description'
        obj.logger = self.fixture.logger
        responses = [http_fixtures.response(content='{"object_id":"4"}')] + [
            http_fixtures.response(content='{"object_id":"18"}') for _ in range(5)]
        with patch.object(chat.requests, 'post', side_effect=responses) as post, \
                patch.object(chat.time, 'sleep'), patch.object(chat.logging, 'warning'), \
                patch.object(ranking.concurrent.futures, 'ThreadPoolExecutor', fixtures.ImmediateExecutor), \
                patch.object(ranking.concurrent.futures, 'as_completed', lambda fs: iter(fs)), \
                redirect_stdout(StringIO()):
            selected = obj.generator_search(['synthetic-image'], [0, 3, 18, 19, 21], 5)
        self.assertEqual(selected, 18)
        self.assertEqual(post.call_count, 6)


if __name__ == '__main__':
    unittest.main()
