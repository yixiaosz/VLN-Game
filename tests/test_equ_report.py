"""Terminal report checks using offline equilibrium traces and crash fragments."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_equ_trace as fixtures
from utils.equ_report import episode_report, write_episode_report
from utils.equ_trace import EquilibriumTraceRun


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TraceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.logger = self.fixture.logger

    def source_hashes(self, root):
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob('*') if path.is_file() and path.name != 'equilibrium.txt'}

    def search(self, ids, generator, discriminator, **kwargs):
        result = self.fixture.run_search(ids, generator, discriminator, **kwargs)
        run = result[-1]
        path = run.root / 'episodes' / '001' / 'equilibrium.jsonl'
        return run, path, path.with_suffix('.txt').read_text()

    def test_auto_report_contains_raw_opinions_and_complete_policies(self):
        run, path, report = self.search([15, 51], [15, 15, -1, 15, 15],
                                        {15: ['correct'] * 4 + ['incorrect']})
        self.assertIn('EPISODE 1 - EQUILIBRIUM SEARCH REPORT', report)
        self.assertIn('COMPLETE POLICY SNAPSHOTS', report)
        for label in ('initial(0)', '125 (25%)', '250 (50%)', '375 (75%)', '500 (100%)'):
            self.assertIn(label, report)
        for header in ('G(correct)', 'G(incorrect)', 'D(correct)', 'D(incorrect)'):
            self.assertIn(header, report)
        self.assertIn('5.739907738e-08', report)
        self.assertIn('"object_id": "15"', report)
        self.assertIn('d-15-5:', report)
        self.assertIn('Discriminator internal key -> candidate ID: {0: 15, -1: -1}', report)
        self.assertIn('Excluded from equilibrium support (zero Generator votes): 51', report)
        self.assertIn('derived_complement', report)
        self.assertIn('no additional VLM queries', report)
        self.assertIn('Selected option: 15', report)
        self.assertIn('not navigation success', report)
        self.assertTrue(all(len(line) <= 100 for line in report.splitlines()))
        before = self.source_hashes(run.root)
        write_episode_report(path)
        self.assertEqual(report, path.with_suffix('.txt').read_text())
        self.assertEqual(before, self.source_hashes(run.root))

    def test_accumulation_negative_values_and_nans_remain_visible(self):
        # Historical rendering must not depend on reproducing the runtime bug.
        path = self.root / 'equilibrium.jsonl'
        fixture = Path(__file__).parent / 'fixtures' / 'legacy_discriminator.jsonl'
        path.write_bytes(fixture.read_bytes())
        (self.root / 'run.json').write_text(json.dumps({
            'run_id': 'synthetic-legacy', 'discriminator_aggregation': 'uncorrected_reference_accumulation'}))
        before = self.source_hashes(self.root)
        report = write_episode_report(path).read_text()
        self.assertIn('REFERENCE ACCUMULATION: candidate 51', report)
        self.assertIn('10 pooled responses, 5 new', report)
        self.assertIn('Correct count used: 6 / denominator 5', report)
        self.assertIn('OUT-OF-RANGE', report)
        self.assertIn('POLICY ANOMALY', report)
        self.assertIn('-0.1363636364', report)
        self.assertIn('NaN', report)
        self.assertIn('queried_all_input_images', report)
        self.assertIn('incorrect responses count toward rejection', report)
        self.assertIn('uncorrected_reference_accumulation', report)
        self.assertEqual(before, self.source_hashes(self.root))

    def test_corrected_runtime_reports_candidate_local_scoring(self):
        _, _, report = self.search([15, 51], [15, 15, 15, 51, 51],
                                   {15: ['correct'] * 4 + ['incorrect'],
                                    51: ['correct'] * 2 + ['incorrect'] * 3,
                                    -1: ['incorrect'] * 5})
        self.assertIn('candidate_local_responses_v1', report)
        self.assertNotIn('REFERENCE ACCUMULATION', report)
        self.assertNotIn('OUT-OF-RANGE', report)
        self.assertNotIn('POLICY ANOMALY', report)
        self.assertIn('Correct count used: 2 / denominator 5', report)

    def test_all_reject_has_no_fabricated_policy_snapshots(self):
        _, _, report = self.search([15], [-1] * 5, {})
        self.assertIn('Early rejection: policies were not initialized', report)
        self.assertIn('Equilibrium updates completed: 0', report)
        self.assertIn('Selected option: -1 (reject)', report)
        self.assertNotIn('initial(0)', report)
        self.assertNotIn('125 (25%)', report)

    def test_failed_samples_show_actual_denominators_and_raw_errors(self):
        _, _, report = self.search(
            [15], [15, 15, RuntimeError('generator failure'), 15, -1],
            {15: ['correct', RuntimeError('discriminator failure'), 'incorrect',
                  'correct', RuntimeError('discriminator failure')]})
        self.assertIn('G=4/5, D=3/5', report)
        self.assertIn('Actual denominator: 4', report)
        self.assertIn('Correct count used: 2 / denominator 5', report)
        self.assertIn('SAMPLE ERROR', report)
        self.assertIn('generator failure', report)
        self.assertIn('discriminator failure', report)

    def test_unexpected_dimensions_and_image_references(self):
        run, path, report = self.search([15, 51], [-1] * 5, {}, sizes={51: (640, 800)})
        self.assertIn('UNEXPECTED IMAGE SIZE: candidate 51', report)
        self.assertIn('640x800', report)
        self.assertIn('search-001-input-001-candidate-51.jpeg', report)
        events = [json.loads(line) for line in path.read_text().splitlines()]
        image = next(e for e in events if e['event'] == 'image_input' and e['candidate_id'] == 51)
        self.assertIn(image['sha256'], report)
        self.assertEqual(hashlib.sha256((run.root / image['archive_path']).read_bytes()).hexdigest(), image['sha256'])

    def test_partial_episode_and_truncated_last_record_are_explicit(self):
        run = EquilibriumTraceRun(self.root, {}, {}, self.logger)
        episode = run.start_episode(1, '0', 'scene', 'instruction')
        episode.start_search(20, [15])
        episode.stream.close()  # Simulate process loss without completion events.
        with episode.path.open('a') as stream:
            stream.write('{"event": "vlm_response", "raw_content":')
        before = self.source_hashes(run.root)
        output = write_episode_report(episode.path)
        report = output.read_text()
        self.assertIn('TRUNCATED final JSONL record', report)
        self.assertIn('episode_end is missing', report)
        self.assertIn('search_end is missing', report)
        self.assertIn('Selected option: not recorded', report)
        self.assertIn('episode outcome is unknown', report)
        self.assertNotIn('Selected option: -1 (reject)', report)
        self.assertEqual(before, self.source_hashes(run.root))

    def test_invalid_middle_record_does_not_overwrite_existing_report(self):
        path = self.root / 'equilibrium.jsonl'
        path.write_text('{"event":"episode_start"}\nnot-json\n{"event":"episode_end"}\n')
        output = path.with_suffix('.txt')
        output.write_text('previous report\n')
        with self.assertRaisesRegex(ValueError, 'line 2'):
            write_episode_report(path)
        self.assertEqual(output.read_text(), 'previous report\n')

    def test_multiple_searches_zero_search_episode_and_cli_conversion(self):
        run = EquilibriumTraceRun(self.root, {}, {}, self.logger)
        episode = run.start_episode(1, '0', 'scene', 'instruction')
        episode.start_search(20, [15]).finish('early_reject', selected_candidate_id=-1)
        episode.start_search(80, [51]).finish('early_reject', selected_candidate_id=-1)
        episode.finish('completed', 100, metrics={'success': 0, 'spl': 0, 'distance_to_goal': 4})
        zero = run.start_episode(2, '1', 'scene', 'instruction')
        zero.finish('interrupted', 0)
        before = self.source_hashes(run.root)
        result = subprocess.run([sys.executable, '-m', 'utils.equ_report', str(run.root), str(episode.path)],
                                cwd=Path(__file__).resolve().parents[1], text=True,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 2)
        report = episode.path.with_suffix('.txt').read_text()
        self.assertIn('SEARCH 1 | agent step 20', report)
        self.assertIn('SEARCH 2 | agent step 80', report)
        self.assertIn('No searches recorded.', zero.path.with_suffix('.txt').read_text())
        self.assertIn('status=interrupted', zero.path.with_suffix('.txt').read_text())
        self.assertEqual(before, self.source_hashes(run.root))

    def test_report_failure_cannot_change_completed_episode(self):
        run = EquilibriumTraceRun(self.root, {}, {}, self.logger)
        episode = run.start_episode(1, '0', 'scene', 'instruction')
        metrics = {'success': 1, 'spl': 0.25, 'distance_to_goal': 0.1}
        with patch('utils.equ_trace.write_episode_report', side_effect=OSError('report unavailable')):
            with self.assertLogs(self.logger, level='ERROR') as captured:
                episode.finish('completed', 100, metrics=metrics)
        self.assertTrue(episode.stream.closed)
        end = json.loads(episode.path.read_text().splitlines()[-1])
        self.assertEqual(end['status'], 'completed')
        self.assertEqual(end['metrics'], metrics)
        self.assertIn('JSONL retained', captured.output[0])

    def test_raw_unicode_controls_and_long_lines_are_terminal_safe(self):
        path = self.root / 'equilibrium.jsonl'
        records = [
            {'event': 'episode_start', 'episode_number': 1, 'instruction': 'x' * 250},
            {'event': 'search_start', 'search_id': 1, 'candidate_order': [15]},
            {'event': 'vlm_response', 'search_id': 1, 'role': 'generator',
             'sample_id': 'g-1', 'completion_order': 1, 'status': 'returned',
             'raw_content': 'line one\n\u4f60\u597d\x1b[31m\tend'},
        ]
        path.write_text(''.join(json.dumps(record) + '\n' for record in records))
        report = episode_report(path)
        self.assertTrue(report.isascii())
        self.assertNotIn('\x1b', report)
        self.assertIn('\\x1b', report)
        self.assertIn('\\u4f60\\u597d', report)
        self.assertTrue(all(len(line) <= 100 for line in report.splitlines()))
        self.assertIn('manifest unavailable', report)


if __name__ == '__main__':
    unittest.main()
