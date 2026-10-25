import os
import numpy as np
import ast
import jsonschema
from jsonschema import validate
from utils.json_validate import schema
import time
import requests
import json
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from arguments import get_args
from urllib.parse import urlsplit, urlunsplit


gpt_name = [
            'text-davinci-003',
            'gpt-3.5-turbo-0125',
            'gpt-4o',
            'gpt-4o-mini'
        ]           

args = get_args()

api_key = os.environ.get("VLN_API_KEY") or args.api
if not api_key or api_key == "xx-xxxx":
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or api_key
    )

api_base = (
    os.environ.get("VLN_API_BASE")
    or os.environ.get("DEEPSEEK_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or "https://api.openai.com/v1"
)

text_model = os.environ.get("VLN_MODEL") or "gpt-4o"

def _env_bool(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}

supports_temperature = _env_bool("VLN_MODEL_SUPPORTS_TEMPERATURE", True)

def _model_options(model_name):
    raw = os.environ.get("VLN_MODEL_OPTIONS")
    if raw:
        try:
            options = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid VLN_MODEL_OPTIONS: {exc}") from exc
        if not isinstance(options, dict):
            raise ValueError("VLN_MODEL_OPTIONS must be a JSON object")
        return options
    if model_name.startswith("deepseek") or "deepseek" in api_base:
        return {"thinking": {"type": "disabled"}}
    return {}

import openai
from openai.error import OpenAIError
openai.api_key = api_key
openai.api_base = api_base


API_TIMEOUT = (10.0, 90.0)  # Connect/read inactivity, not a wall-clock deadline.
API_MAX_ATTEMPTS = 3
API_RETRY_DELAYS = (2.0, 4.0)
API_RETRY_AFTER_CAP = 30.0
API_BODY_PREVIEW_LIMIT = 2048


class APIRequestError(RuntimeError):
    """A required logical sample failed; do not aggregate a partial batch."""


class _ResponseError(ValueError):
    def __init__(self, category):
        super().__init__(category)
        self.category = category


def _safe_preview(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    text = str(value or '')
    if api_key and api_key != 'xx-xxxx':
        text = text.replace(api_key, '<redacted>')
    text = re.sub(r'(?i)Bearer\s+[^\s"\']+', 'Bearer <redacted>', text)
    text = re.sub(r'\bsk-[A-Za-z0-9_-]+', '<redacted>', text)
    text = re.sub(r'(https?://)[^/\s@]+@', r'\1<redacted>@', text)
    text = re.sub(r'data:image/[^;,]+;base64,[A-Za-z0-9+/=]+', '<image omitted>', text)
    return text[:API_BODY_PREVIEW_LIMIT]


def _http_details(status, headers):
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    headers = {key.lower(): value for key, value in (headers or {}).items()}
    return {'http_status': status, 'content_type': _safe_preview(headers.get('content-type')),
            'request_id': _safe_preview(headers.get('x-request-id') or
                                        headers.get('x-openrouter-request-id')),
            'retry_after': _safe_preview(headers.get('retry-after'))}


def _response_content(data, details):
    if isinstance(data, dict) and 'error' in data:
        error = data['error']
        code = error.get('code') if isinstance(error, dict) else None
        details['provider_error_code'] = code if isinstance(code, (int, float, type(None))) else _safe_preview(code)
        raise _ResponseError('provider_error')
    try:
        content = data['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
        raise _ResponseError('response_envelope') from None
    if not isinstance(content, str) or not content.strip():
        raise _ResponseError('empty_content')
    return content


def _retry_delay(attempt, details):
    delay = API_RETRY_DELAYS[min(attempt - 1, len(API_RETRY_DELAYS) - 1)]
    value = details.get('retry_after')
    if value:
        try:
            seconds = float(value)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                seconds = 0
        delay = max(delay, min(API_RETRY_AFTER_CAP, max(0, seconds)))
    return delay


def _run_chat_request(request, model, role, json_mode, attempt_observer, cancel_event=None):
    for attempt in range(1, API_MAX_ATTEMPTS + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise APIRequestError(f'{role} API retries cancelled after batch interruption/failure')
        started = time.monotonic()
        details = {'http_status': None, 'model': model}
        if attempt_observer:
            attempt_observer('api_attempt_start', role=role, attempt=attempt, model=model,
                             connect_timeout_seconds=API_TIMEOUT[0], read_timeout_seconds=API_TIMEOUT[1])
        try:
            content = request(details)
            if json_mode:
                try:
                    ast.literal_eval(content)  # Preserve the existing content parser.
                except (SyntaxError, ValueError):
                    details['model_content_preview'] = _safe_preview(content)
                    raise _ResponseError('model_content') from None
        except (requests.RequestException, OpenAIError, ValueError) as exc:
            if isinstance(exc, OpenAIError):
                details.update(_http_details(getattr(exc, 'http_status', None),
                                             getattr(exc, 'http_headers', None)))
                body = getattr(exc, 'http_body', None)
                if body is not None:
                    details['body_preview'] = _safe_preview(body)
            if isinstance(exc, _ResponseError):
                category = exc.category
            elif isinstance(exc, requests.exceptions.JSONDecodeError):
                category = 'response_json'
            elif isinstance(exc, requests.Timeout) or type(exc).__name__ == 'Timeout':
                category = 'timeout'
            elif isinstance(exc, requests.ConnectionError):
                category = 'connection'
            elif isinstance(exc, ValueError):
                category = 'configuration'
            else:
                category = 'http_or_sdk_error'
            status = details.get('http_status')
            code = details.get('provider_error_code')
            permanent_code = code in ('invalid_api_key', 'model_not_found', 'invalid_request_error',
                                      'insufficient_quota', 'unsupported_parameter')
            permanent_sdk_error = type(exc).__name__ in ('AuthenticationError', 'InvalidRequestError', 'PermissionError')
            try:
                code = int(code)
            except (TypeError, ValueError):
                code = None
            effective_status = code if code is not None else status
            transient_status = (effective_status is None or 200 <= effective_status < 300 or
                                effective_status in (408, 429, 500, 502, 503, 504))
            retryable = category != 'configuration' and not permanent_code and not permanent_sdk_error and transient_status
            retry = retryable and attempt < API_MAX_ATTEMPTS
            delay = _retry_delay(attempt, details) if retry else 0
            if attempt_observer:
                attempt_observer('api_attempt_end', role=role, attempt=attempt,
                                 status='retry' if retry else 'failed', error_type=type(exc).__name__,
                                 error_category=category, retryable=retryable,
                                 retry_delay_seconds=delay, elapsed_seconds=time.monotonic() - started,
                                 **details)
            logging.warning('API %s attempt %d/%d: %s, HTTP=%s, retry=%s',
                            role, attempt, API_MAX_ATTEMPTS, category, status, retry)
            if not retry:
                raise APIRequestError(f'{role} API failed after {attempt} application attempt(s): '
                                      f'{category} [{type(exc).__name__}], HTTP={status}') from None
            if cancel_event is None:
                time.sleep(delay)
            elif cancel_event.wait(delay):
                raise APIRequestError(f'{role} API retries cancelled after batch interruption/failure')
        else:
            if attempt_observer:
                attempt_observer('api_attempt_end', role=role, attempt=attempt, status='returned',
                                 elapsed_seconds=time.monotonic() - started, **details)
            print(model + ' response: ')
            print(content)
            return content


def trace_configuration():
    """Non-secret effective request settings, for the structured run manifest."""
    def redact(value):
        if isinstance(value, dict):
            return {key: ('<redacted>' if key.lower() in {
                'api', 'api_key', 'authorization', 'password', 'secret', 'token',
                'access_token', 'api_token'} else redact(item))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    vision_model = os.environ.get("VLN_MODEL") or gpt_name[args.gpt_type]
    configurations = {}
    for role, model, limit, temperature in (
            ('parsing', text_model, 512, 0.5),
            ('generator', vision_model, 100, 1.1),
            ('discriminator', vision_model, 100, 1.1)):
        options = _model_options(model).copy()
        options.setdefault('max_tokens', limit)
        if supports_temperature:
            options.setdefault('temperature', temperature)
        else:
            options.pop('temperature', None)
        options['model'] = model
        options.pop('messages', None)
        options.pop('response_format', None)
        if role != 'discriminator':
            options['response_format'] = {'type': 'json_object'}
        configurations[role] = redact(options)
    # The parsing request forces 512 tokens only before applying model options,
    # just as chat_with_gpt does. No authorization headers or keys are included.
    url = urlsplit(api_base)
    safe_base = urlunsplit((url.scheme, url.netloc.rsplit('@', 1)[-1], url.path, '', ''))
    return {'api_base': safe_base, 'supports_temperature': supports_temperature,
            'requests': configurations,
            'transport': {'protocol': 'http_timeouts_bounded_attempts_v1',
                          'connect_timeout_seconds': API_TIMEOUT[0], 'read_timeout_seconds': API_TIMEOUT[1],
                          'max_application_attempts': API_MAX_ATTEMPTS,
                          'retry_delays_seconds': list(API_RETRY_DELAYS),
                          'retry_after_cap_seconds': API_RETRY_AFTER_CAP,
                          'timeout_kind': 'connect/read inactivity; not an absolute wall-clock deadline',
                          'sample_failure_policy': 'abort_test_no_partial_batch'}}



def chat_with_gpt(chat_history, attempt_observer=None, cancel_event=None):
    def request(details):
        params = {
            "model": text_model,
            "response_format": {"type": "json_object"},
            "messages": chat_history,
            "max_tokens": 512,
        }
        params.update(_model_options(text_model))
        params["model"] = text_model
        params["messages"] = chat_history
        params["response_format"] = {"type": "json_object"}
        if supports_temperature:
            params.setdefault("temperature", 0.5)
        else:
            params.pop("temperature", None)
        params['request_timeout'] = API_TIMEOUT
        response = openai.ChatCompletion.create(**params)
        try:
            content = response.choices[0].message['content']
        except (AttributeError, KeyError, IndexError, TypeError):
            details['body_preview'] = _safe_preview(response)
            raise _ResponseError('response_envelope') from None
        if not isinstance(content, str) or not content.strip():
            raise _ResponseError('empty_content')
        return content
    return _run_chat_request(request, text_model, 'parsing', True, attempt_observer, cancel_event)

def chat_with_gpt4v(chat_history, gpt_type = args.gpt_type, json_mode = True,
                   attempt_observer=None, cancel_event=None):
    
    model_name = os.environ.get("VLN_MODEL") or gpt_name[gpt_type]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    def request(details):
        payload = {
            "model": model_name,
            "messages": chat_history,
            "max_tokens": 100,
        }
        payload.update(_model_options(model_name))
        payload["model"] = model_name
        payload["messages"] = chat_history
        payload.setdefault("max_tokens", 100)
        if supports_temperature:
            payload.setdefault("temperature", 1.1)
        else:
            payload.pop("temperature", None)
        if json_mode:
            payload["response_format"] = { "type": "json_object" }
        else:
            payload.pop("response_format", None)

        response = requests.post(api_base.rstrip("/") + "/chat/completions", headers=headers,
                                 json=payload, timeout=API_TIMEOUT)
        try:
            details.update(_http_details(response.status_code, response.headers))
            response.raise_for_status()
            return _response_content(response.json(), details)
        except (requests.RequestException, _ResponseError):
            details['body_preview'] = _safe_preview(response.text)
            details['body_preview_truncated'] = len(response.text) > API_BODY_PREVIEW_LIMIT
            raise
        finally:
            response.close()
    return _run_chat_request(request, model_name, 'generator' if json_mode else 'discriminator',
                             json_mode, attempt_observer, cancel_event)

def chat_with_llama3(user_prompt, chat_history):

    chat_history.append({"role": "user", "content": user_prompt})

    retries = 10    
    while retries > 0:  
        
        url = 'http://localhost:11434/api/chat'
        data = {
            "model": "llama3",
            "messages": chat_history,
            "stream": False,
            "format": "json",
            "options": {
                "seed": 101,
                "temperature": 0
            }
        }
        
        response = requests.post(url, json=data)
        decoded_string = response.content.decode('utf-8')
        response_message = json.loads(decoded_string)["message"]["content"]
        ground_json = ast.literal_eval(response_message)
                
        try: 
            validate(instance = ground_json, schema = schema)
        except jsonschema.exceptions.ValidationError as err:
            print("JSON data is invalid", err)
            print('Json format error, retrying...')    
            retries -= 1

            
    chat_history.append({"role": "assistant", "content": response_message})
    print(response_message)
    return response_message
