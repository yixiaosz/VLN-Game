"""Bounded API failures: mocks plus a local stalled HTTP server; no billed API calls."""
import ast
from collections import defaultdict
from contextlib import ExitStack, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
import logging
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

with patch.object(sys, 'argv', [sys.argv[0]]):
    import utils.chat_utils as chat
from utils.candidate_images import CandidateRenderCancelled
from utils.equ_trace import EquilibriumTraceRun
import utils.equ_ranking as ranking


def response(status=200, content='{"object_id":"15"}', body=None, headers=None):
    result = requests.Response()
    result.status_code = status
    result.url = 'https://unit.example/v1/chat/completions'
    result.headers.update({'Content-Type': 'application/json', 'X-Request-ID': 'unit-request'})
    result.headers.update(headers or {})
    result._content = (json.dumps({'choices': [{'message': {'content': content}}]}) if body is None else body).encode()
    result._content_consumed = True
    return result


class APITests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(chat.os.environ, {'VLN_MODEL': 'vision-model', 'VLN_MODEL_OPTIONS': ''}))
        self.stack.enter_context(patch.object(chat, 'text_model', 'parsing-model'))
        self.stack.enter_context(patch.object(chat, 'api_base', 'https://unit.example/v1'))
        self.stack.enter_context(patch.object(chat, 'api_key', 'test-secret-value'))
        self.stack.enter_context(patch.object(chat, 'supports_temperature', True))
        self.stack.enter_context(patch.object(chat.logging, 'warning'))
        self.events = []
        self.observer = lambda event, **fields: self.events.append(dict(event=event, **fields))
        self.history = [{'role': 'user', 'content': 'unit prompt'}]

    def invoke(self, **kwargs):
        with redirect_stdout(StringIO()):
            return chat.chat_with_gpt4v(self.history, attempt_observer=self.observer, **kwargs)

    def test_success_preserves_payload_and_adds_timeout(self):
        with patch.object(chat.requests, 'post', return_value=response()) as post:
            self.assertEqual(self.invoke(), '{"object_id":"15"}')
        self.assertEqual(post.call_count, 1)
        args, kwargs = post.call_args
        self.assertEqual(args, ('https://unit.example/v1/chat/completions',))
        self.assertEqual(kwargs['timeout'], (10.0, 90.0))
        self.assertEqual(kwargs['json'], {'model': 'vision-model', 'messages': self.history,
                                         'max_tokens': 100, 'temperature': 1.1,
                                         'response_format': {'type': 'json_object'}})
        self.assertNotIn('Authorization', json.dumps(self.events))
        self.assertNotIn('test-secret-value', json.dumps(self.events))
        self.assertEqual(self.events[-1]['status'], 'returned')

    def test_non_json_envelope_retries_then_returns_one_logical_result(self):
        with patch.object(chat.requests, 'post', side_effect=[response(body=''), response()]) as post, \
                patch.object(chat.time, 'sleep') as sleep:
            self.assertEqual(self.invoke(), '{"object_id":"15"}')
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(2.0)
        failed = next(e for e in self.events if e.get('status') == 'retry')
        self.assertEqual(failed['error_category'], 'response_json')
        self.assertEqual(failed['http_status'], 200)
        self.assertEqual(failed['request_id'], 'unit-request')
        self.assertEqual(failed['body_preview'], '')

    def test_http_status_is_checked_before_json(self):
        bad = response(502, body='<html>Bad gateway</html>', headers={'Content-Type': 'text/html'})
        with patch.object(chat.requests, 'post', side_effect=[bad, response()]) as post, \
                patch.object(chat.time, 'sleep'):
            self.invoke()
        self.assertEqual(post.call_count, 2)
        failed = next(e for e in self.events if e.get('status') == 'retry')
        self.assertEqual(failed['http_status'], 502)
        self.assertEqual(failed['error_type'], 'HTTPError')
        self.assertEqual(failed['content_type'], 'text/html')
        self.assertIn('Bad gateway', failed['body_preview'])

    def test_permanent_http_error_fails_without_retry_and_redacts_body(self):
        bad = response(401, body='Invalid key test-secret-value Bearer test-secret-value ' + 'x' * 3000)
        with patch.object(chat.requests, 'post', return_value=bad) as post, patch.object(chat.time, 'sleep') as sleep:
            with self.assertRaises(chat.APIRequestError) as caught:
                self.invoke()
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()
        text = json.dumps(self.events) + str(caught.exception)
        self.assertNotIn('test-secret-value', text)
        failed = self.events[-1]
        self.assertLessEqual(len(failed['body_preview']), chat.API_BODY_PREVIEW_LIMIT)
        self.assertTrue(failed['body_preview_truncated'])

    def test_invalid_model_syntax_consumes_all_three_attempts(self):
        with patch.object(chat.requests, 'post', side_effect=lambda *a, **k: response(content='not JSON')) as post, \
                patch.object(chat.time, 'sleep'):
            with self.assertRaises(chat.APIRequestError):
                self.invoke()
        self.assertEqual(post.call_count, 3)
        self.assertEqual([e['attempt'] for e in self.events if e['event'] == 'api_attempt_start'], [1, 2, 3])
        self.assertEqual(self.events[-1]['error_category'], 'model_content')

    def test_empty_content_and_missing_choices_are_bounded(self):
        for body in ('{"choices":[]}', '{"choices":[{"message":{"content":""}}]}'):
            with self.subTest(body=body):
                with patch.object(chat.requests, 'post', side_effect=lambda *a, **k: response(body=body)) as post, \
                        patch.object(chat.time, 'sleep'):
                    with self.assertRaises(chat.APIRequestError):
                        self.invoke()
                self.assertEqual(post.call_count, 3)

    def test_http_200_error_envelope_is_not_an_unexplained_choices_keyerror(self):
        body = '{"error":{"code":400,"message":"unsupported parameter"}}'
        with patch.object(chat.requests, 'post', return_value=response(body=body)) as post:
            with self.assertRaises(chat.APIRequestError):
                self.invoke()
        self.assertEqual(post.call_count, 1)
        self.assertEqual(self.events[-1]['error_category'], 'provider_error')
        self.assertEqual(self.events[-1]['provider_error_code'], 400)
        with patch.object(chat.requests, 'post', side_effect=lambda *a, **k: response(
                body='{"error":{"code":"test-secret-value"}}')), patch.object(chat.time, 'sleep'):
            with self.assertRaises(chat.APIRequestError):
                self.invoke()
        self.assertNotIn('test-secret-value', json.dumps(self.events))

    def test_retry_after_is_respected_with_a_cap(self):
        with patch.object(chat.requests, 'post', side_effect=[response(429, headers={'Retry-After': '9999'}), response()]), \
                patch.object(chat.time, 'sleep') as sleep:
            self.invoke()
        sleep.assert_called_once_with(30.0)

    def test_timeout_exhaustion_is_explicit(self):
        with patch.object(chat.requests, 'post', side_effect=requests.ReadTimeout('unit socket timeout')) as post, \
                patch.object(chat.time, 'sleep'):
            with self.assertRaisesRegex(chat.APIRequestError, '3 application attempt'):
                self.invoke()
        self.assertEqual(post.call_count, 3)
        self.assertEqual(self.events[-1]['error_category'], 'timeout')

    def test_parsing_sdk_timeout_and_payload_are_preserved(self):
        reply = SimpleNamespace(choices=[SimpleNamespace(message={'content': '{"target":"table"}'})])
        with patch.object(chat.openai.ChatCompletion, 'create', return_value=reply) as create, redirect_stdout(StringIO()):
            self.assertEqual(chat.chat_with_gpt(self.history, attempt_observer=self.observer), '{"target":"table"}')
        self.assertEqual(create.call_args.kwargs, {'model': 'parsing-model', 'messages': self.history,
                         'max_tokens': 512, 'temperature': 0.5, 'response_format': {'type': 'json_object'},
                         'request_timeout': (10.0, 90.0)})

    def test_parsing_malformed_content_is_bounded(self):
        reply = SimpleNamespace(choices=[SimpleNamespace(message={'content': 'not JSON'})])
        with patch.object(chat.openai.ChatCompletion, 'create', return_value=reply) as create, \
                patch.object(chat.time, 'sleep'), redirect_stdout(StringIO()):
            with self.assertRaises(chat.APIRequestError):
                chat.chat_with_gpt(self.history, attempt_observer=self.observer)
        self.assertEqual(create.call_count, 3)

    def test_sdk_authentication_error_without_status_is_not_retried(self):
        error = chat.openai.error.AuthenticationError('invalid test-secret-value')
        with patch.object(chat.openai.ChatCompletion, 'create', side_effect=error) as create, \
                patch.object(chat.time, 'sleep') as sleep, redirect_stdout(StringIO()):
            with self.assertRaisesRegex(chat.APIRequestError, 'AuthenticationError') as caught:
                chat.chat_with_gpt(self.history, attempt_observer=self.observer)
        self.assertEqual(create.call_count, 1)
        sleep.assert_not_called()
        self.assertNotIn('test-secret-value', str(caught.exception) + json.dumps(self.events))

    def test_episode_71_parsing_structure_retries_same_request_without_repair(self):
        # Exact raw content from the stopped run, including its whitespace.
        bad = '{\n    "target": "couch",\n    "landmark": [\n        "table",\n        "flower vase",\n    []\n]  \n}'
        good = '{"target":"couch","landmark":["table","flower vase"]}'
        replies = [SimpleNamespace(choices=[SimpleNamespace(message={'content': text})]) for text in (bad, good)]
        with patch.object(chat.openai.ChatCompletion, 'create', side_effect=replies) as create, \
                patch.object(chat.time, 'sleep') as sleep, redirect_stdout(StringIO()):
            self.assertEqual(chat.chat_with_gpt(self.history, attempt_observer=self.observer), good)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args_list[0], create.call_args_list[1])
        sleep.assert_called_once_with(2.)
        retry = next(e for e in self.events if e.get('status') == 'retry')
        self.assertEqual(retry['error_category'], 'parsing_structure')
        self.assertEqual(retry['model_content_preview'], bad)
        self.assertEqual(self.events[-1]['status'], 'returned')

    def test_invalid_parsing_structures_exhaust_existing_attempt_budget(self):
        invalid = ['[]', '{}', '{"target":None}', '{"target":4}', '{"target":[]}',
                   '{"target":"couch","landmark":"table"}',
                   '{"target":"couch","landmark":None}',
                   '{"target":"couch","landmark":{}}',
                   '{"target":"couch","landmark":["table",[]]}',
                   '{"target":"couch","landmark":[4]}']
        for content in invalid:
            with self.subTest(content=content):
                self.events.clear()
                reply = SimpleNamespace(choices=[SimpleNamespace(message={'content': content})])
                with patch.object(chat.openai.ChatCompletion, 'create', return_value=reply) as create, \
                        patch.object(chat.time, 'sleep') as sleep, redirect_stdout(StringIO()):
                    with self.assertRaisesRegex(chat.APIRequestError, '3 application attempt.*parsing_structure'):
                        chat.chat_with_gpt(self.history, attempt_observer=self.observer)
                self.assertEqual(create.call_count, 3)
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [2., 4.])
                self.assertEqual(self.events[-1]['status'], 'failed')

    def test_valid_parsing_shapes_return_original_content_without_extra_restrictions(self):
        valid = ['{"target":"couch"}', '{"target":"couch","landmark":[]}',
                 '{"target":"couch","landmark":["table","flower vase"]}',
                 '{"target":"couch","landmark":[""],"extra":[]}', '{"target":""}']
        for content in valid:
            with self.subTest(content=content):
                reply = SimpleNamespace(choices=[SimpleNamespace(message={'content': content})])
                with patch.object(chat.openai.ChatCompletion, 'create', return_value=reply) as create, \
                        patch.object(chat.time, 'sleep') as sleep, redirect_stdout(StringIO()):
                    self.assertEqual(chat.chat_with_gpt(self.history), content)
                create.assert_called_once()
                sleep.assert_not_called()

    def test_parsing_structure_validation_does_not_change_generator_validation(self):
        # Generator retains syntax-only validation; downstream object_id semantics
        # are a separate decision from the approved parsing-only correction.
        with patch.object(chat.requests, 'post', return_value=response(content='[]')) as post:
            self.assertEqual(self.invoke(), '[]')
        post.assert_called_once()

    def test_discriminator_words_and_model_options_keep_existing_semantics(self):
        with patch.dict(chat.os.environ, {'VLN_MODEL_OPTIONS': '{"max_tokens":222,"temperature":0.7}'}), \
                patch.object(chat.requests, 'post', return_value=response(content='maybe')) as post:
            self.assertEqual(self.invoke(json_mode=False), 'maybe')
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['max_tokens'], 222)
        self.assertEqual(payload['temperature'], 0.7)
        self.assertNotIn('response_format', payload)

    def test_cancelled_batch_does_not_start_more_http_attempts(self):
        cancel = threading.Event()
        cancel.set()
        with patch.object(chat.requests, 'post') as post:
            with self.assertRaisesRegex(chat.APIRequestError, 'cancelled'):
                self.invoke(cancel_event=cancel)
        post.assert_not_called()

    def test_invalid_model_configuration_fails_before_http_without_retry(self):
        with patch.dict(chat.os.environ, {'VLN_MODEL_OPTIONS': 'not JSON'}), \
                patch.object(chat.requests, 'post') as post, patch.object(chat.time, 'sleep') as sleep:
            with self.assertRaises(chat.APIRequestError):
                self.invoke()
        post.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(self.events[-1]['error_category'], 'configuration')

    def test_real_http_read_timeout_returns_without_a_billed_api(self):
        class Handler(BaseHTTPRequestHandler):
            count = 0
            def do_POST(self):
                type(self).count += 1
                self.rfile.read(int(self.headers.get('Content-Length', 0)))
                time.sleep(0.2)
                try:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{}')
                except OSError:
                    pass
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        try:
            with patch.object(chat, 'api_base', f'http://127.0.0.1:{server.server_port}'), \
                    patch.object(chat, 'API_TIMEOUT', (0.1, 0.03)), patch.object(chat, 'API_RETRY_DELAYS', (0., 0.)):
                started = time.monotonic()
                with self.assertRaises(chat.APIRequestError):
                    self.invoke()
                self.assertLess(time.monotonic() - started, 1.0)
                self.assertEqual(Handler.count, 3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(1)

    def test_interrupt_stops_peer_retries_before_executor_shutdown(self):
        started = threading.Event()
        lock = threading.Lock()
        calls = []
        def blocked_request(*args, **kwargs):
            with lock:
                calls.append(kwargs['timeout'])
                if len(calls) == 5:
                    started.set()
            threading.Event().wait(0.1)
            raise requests.ReadTimeout('unit pending request')
        def interrupt(futures):
            self.assertTrue(started.wait(1))
            raise KeyboardInterrupt
        obj = ranking.Equilibrium_Ranking.__new__(ranking.Equilibrium_Ranking)
        obj.trace = None
        obj.candidate_id = [15]
        obj.logger = Mock()
        with patch.object(chat.requests, 'post', side_effect=blocked_request), \
                patch.object(ranking.concurrent.futures, 'as_completed', side_effect=interrupt), \
                redirect_stdout(StringIO()):
            before = time.monotonic()
            with self.assertRaises(KeyboardInterrupt):
                obj._get_generative_probabilities(self.history)
            self.assertLess(time.monotonic() - before, 1)
        self.assertEqual(len(calls), 5, 'Interrupted peer calls must not start additional attempts')

    def test_attempt_diagnostics_are_flushed_and_visible_in_terminal_report(self):
        with tempfile.TemporaryDirectory() as directory:
            run = EquilibriumTraceRun(directory, {'api_configuration': chat.trace_configuration()}, {}, Mock())
            episode = run.start_episode(1, 'unit', 'unit-scene', 'unit retry example')
            search = episode.start_search(20, [15])
            observer = lambda event, **fields: search.emit(event, sample_id='g-1', **fields)
            with patch.object(chat.requests, 'post', side_effect=[response(502, body='gateway unavailable'), response()]), \
                    patch.object(chat.time, 'sleep'), redirect_stdout(StringIO()):
                content = chat.chat_with_gpt4v(self.history, attempt_observer=observer)
            search.emit('vlm_response', role='generator', sample_id='g-1', status='returned', raw_content=content)
            search.finish('completed', selected_candidate_id=15)
            episode.finish('completed', 20)
            events = [json.loads(line) for line in episode.path.read_text().splitlines()]
            self.assertEqual(len([e for e in events if e['event'] == 'api_attempt_start']), 2)
            report = episode.path.with_suffix('.txt').read_text()
            self.assertIn('Recorded application-level API attempts: 2', report)
            self.assertIn('gateway unavailable', report)
            self.assertIn('unit-request', report)
            self.assertNotIn('test-secret-value', report)

    def test_terminal_error_preserves_completed_episode_and_does_not_count_incomplete_one(self):
        # Execute the actual main function with a tiny fake environment, without
        # importing/initializing Habitat, CUDA models, or an Open3D window.
        source = ast.parse(Path('main_vln.py').read_text())
        main_node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        env = SimpleNamespace(episodes=[0, 1], episode_over=False, number=0,
                              sim=SimpleNamespace(get_agent_state=lambda: None))
        env._sim = env.sim
        def reset():
            env.number += 1
            env.episode_over = False
            env.current_episode = SimpleNamespace(episode_id=str(env.number), scene_id='unit-scene')
            return {'instruction': {'text': 'unit instruction'}, 'rgb': np.zeros((2, 2, 3), dtype=np.uint8)}
        env.reset = reset
        env.get_metrics = lambda: {'success': 1., 'spl': .25, 'distance_to_goal': .1}
        env.step = lambda action: setattr(env, 'episode_over', True)
        agent = SimpleNamespace(reset=lambda: None)
        def act(*args):
            if env.number == 1:
                return 0
            agent.episode_trace.start_search(20, [15])
            raise chat.APIRequestError('unit required sample exhausted')
        agent.act = act
        config = SimpleNamespace(SEED=100, defrost=lambda: None, freeze=lambda: None,
                 SIMULATOR=SimpleNamespace(HABITAT_SIM_V0=SimpleNamespace(GPU_DEVICE_ID=0)),
                 ENVIRONMENT=SimpleNamespace(MAX_EPISODE_STEPS=500))
        log = Mock()
        namespace = dict(os=chat.os, logging=log, get_config=lambda **k: config,
                         random=SimpleNamespace(seed=Mock()), np=np,
                         torch=SimpleNamespace(manual_seed=Mock(), set_grad_enabled=Mock()),
                         Env=lambda **k: env, ShortestPathFollowerCompat=lambda *a: None,
                         VLObjectNav_Agent=lambda *a, **k: agent, defaultdict=defaultdict, time=time,
                         EquilibriumTraceRun=EquilibriumTraceRun, trace_configuration=lambda: {},
                         Parsing_prompt='P', generator_prompt='G', discriminator_prompt='D',
                         CANDIDATE_IMAGE_PROTOCOL='unit', RENDER_TIMEOUT_SECONDS=120,
                         transform_rgb_bgr=lambda image: image,
                         cv2=SimpleNamespace(cvtColor=lambda image, mode: image, COLOR_BGR2RGB=0),
                         CandidateRenderCancelled=CandidateRenderCancelled)
        exec(compile(ast.Module(body=[main_node], type_ignores=[]), 'main_error_path', 'exec'), namespace)
        before_rng = np.random.get_state()
        self.addCleanup(np.random.set_state, before_rng)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            args = SimpleNamespace(vln_mode='vlm_game', gpt_type=2, dump_location=directory,
                                    gpu_id=0, seed=1, visualize=1, episode_count=2, gui_closed=False)
            channel = SimpleNamespace(request_gui_close=Mock())
            with self.assertRaises(chat.APIRequestError):
                namespace['main'](args, None, None, channel)
            channel.request_gui_close.assert_called_once_with()
            paths = sorted(Path(directory).glob('logs/*/traces/*/episodes/*/equilibrium.jsonl'))
            ends = [json.loads(path.read_text().splitlines()[-1]) for path in paths]
            self.assertEqual([end['status'] for end in ends], ['completed', 'error'])
            self.assertEqual(ends[0]['metrics']['success'], 1.)
            self.assertIsNone(ends[1]['metrics'])
            logged = str(log.info.call_args_list)
            self.assertIn('---(1/2)', logged)
            self.assertNotIn('---(2/2)', logged)


if __name__ == '__main__':
    unittest.main()
