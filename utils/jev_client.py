"""Native OpenRouter Decisions transport, isolated from chat-model payloads."""
import base64
import hashlib
import json
import math
import os

import requests

from utils.chat_utils import (
    API_TIMEOUT, APIRequestError, _ResponseError, _http_details, _run_chat_request, _safe_preview)

MODEL = 'typesafe/jev-1.14'
ENDPOINT = 'https://openrouter.ai/api/alpha/decisions'
INTRO = ("Each composite contains a first-view image and a top-view point-cloud map. "
         "The red bounding box at the map center highlights the candidate; the top-left red number is its object ID. "
         "Evaluate the candidate attributes and spatial relationships against the full description.")
CHOICE_INSTRUCTION = ('Which candidate best matches the described target, including its attributes and spatial relationships? '
                      'Select -1 if none matches.')
MATCH_INSTRUCTION = ('Does the highlighted candidate match the described target, including all required attributes and spatial relationships?')
REJECT_INSTRUCTION = ('Do none of the offered candidates match the described target, including the required attributes and spatial relationships?')
PROTOCOL = 'jev_decisions_complete_probabilities_v1'


def build_request(instruction, ids, images, kind):
    if len(ids) != len(images) or not ids or len(set(ids)) != len(ids) or -1 in ids:
        raise ValueError('Jev requires distinct real candidate IDs with exactly one image each')
    state = [INTRO, 'Original object description: ' + instruction]
    for cid, encoded in zip(ids, images):
        state.extend([f'Candidate object_id {cid}:',
                      {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + encoded}}])
    if kind == 'choice':
        criteria = {str(cid): f'The highlighted candidate with object_id {cid} matches the description.' for cid in ids}
        criteria['-1'] = 'None of the offered candidates matches the description.'
        question = {'type': 'choice', 'instructions': CHOICE_INSTRUCTION, 'criteria': criteria}
    elif kind == 'match' and len(ids) == 1:
        question = {'type': 'noul', 'instructions': MATCH_INSTRUCTION,
                    'criteria': {'true': 'The highlighted candidate matches the described target and required attributes and spatial relationships.',
                                 'false': 'The highlighted candidate does not match the described target or a required attribute or spatial relationship.'}}
    elif kind == 'reject':
        question = {'type': 'noul', 'instructions': REJECT_INSTRUCTION,
                    'criteria': {'true': 'None of the offered candidates matches the full description.',
                                 'false': 'At least one offered candidate matches the full description.'}}
    else:
        raise ValueError('Unsupported Jev question/scope')
    return {'model': MODEL, 'state': state, 'questions': {'decision': question}}


def probability(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def validate_response(data, payload):
    """Validate full support; only unit-mass floating-point roundoff is normalized."""
    question = payload['questions']['decision']
    try:
        if not isinstance(data['model'], str) or not data['model'].startswith(MODEL + '-') and data['model'] != MODEL:
            raise ValueError('Unexpected resolved Jev model')
        if set(data['answers']) != {'decision'}:
            raise ValueError('Unexpected/missing question answer')
        answer = data['answers']['decision']
        if answer['type'] != question['type']:
            raise ValueError('Wrong answer type')
        if question['type'] == 'choice':
            probs = answer['probabilities']
            if set(probs) != set(question['criteria']) or not all(probability(v) for v in probs.values()):
                raise ValueError('Incomplete or invalid Choice probabilities')
            mass = sum(probs.values())
            if abs(mass - 1) > 1e-6:
                raise ValueError('Choice probability mass differs from one by more than 1e-6')
            if answer['choice'] not in probs or probs[answer['choice']] != max(probs.values()):
                raise ValueError('Choice is not a maximum-probability offered option')
            if 'confidence' in answer and not probability(answer['confidence']):
                raise ValueError('Invalid Choice confidence')
            return {'probabilities': {key: probs[key] / mass for key in question['criteria']},
                    'raw_probability_sum': mass}
        if not probability(answer['noul']):
            raise ValueError('Invalid Noul probability')
        return {'noul': answer['noul']}
    except (KeyError, TypeError, AttributeError, ValueError):
        raise _ResponseError('jev_response_structure') from None


def request_decision(payload, role, observer=None, cancel_event=None):
    key = os.environ.get('VLN_API_KEY') or os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise APIRequestError('OpenRouter API key is not configured for Jev')
    body = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    request_hash = hashlib.sha256(body).hexdigest()
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
               'TypeSafe-Danger-Zone-Images': 'true', 'TypeSafe-Danger-Zone-Long-Context': 'true'}
    if observer:
        observer('jev_request', protocol=PROTOCOL, endpoint=ENDPOINT,
                 request_sha256=request_hash, request_bytes=len(body), questions=payload['questions'],
                 image_sha256=[hashlib.sha256(base64.b64decode(part['image_url']['url'].split(',', 1)[1])).hexdigest()
                               for part in payload['state'] if isinstance(part, dict)],
                 state_text=[part for part in payload['state'] if isinstance(part, str)])
    def request(details):
        details.update(protocol=PROTOCOL, request_sha256=request_hash)
        response = requests.post(ENDPOINT, headers=headers, data=body, timeout=API_TIMEOUT)
        try:
            details.update(_http_details(response.status_code, response.headers))
            response.raise_for_status()
            safe_text = response.text.replace(key, '<redacted>')
            # Preserve every HTTP-success provider envelope, including malformed answers.
            if observer:
                observer('jev_raw_response', raw_content=safe_text, request_sha256=request_hash)
            try:
                data = json.loads(safe_text)
            except json.JSONDecodeError:
                raise _ResponseError('response_json') from None
            if isinstance(data, dict) and 'error' in data:
                error = data['error']
                details['provider_error_code'] = error.get('code') if isinstance(error, dict) else None
                raise _ResponseError('provider_error')
            validate_response(data, payload)
            return safe_text
        except (requests.RequestException, _ResponseError):
            details['body_preview'] = _safe_preview(response.text).replace(key, '<redacted>')
            raise
        finally:
            response.close()
    raw = _run_chat_request(request, MODEL, role, False, observer, cancel_event)
    data = json.loads(raw)
    return data, validate_response(data, payload)


def configuration():
    return {'api_base': ENDPOINT, 'model': MODEL, 'protocol': PROTOCOL,
            'temperature': 'unsupported_omitted', 'choice_mass_tolerance': 1e-6,
            'probability_validation': 'complete_finite_unit_interval_roundoff_mass_only_v1',
            'transport': {'connect_timeout_seconds': API_TIMEOUT[0], 'read_timeout_seconds': API_TIMEOUT[1],
                          'max_application_attempts': 3, 'retry_delays_seconds': [2, 4],
                          'retry_after_cap_seconds': 30,
                          'sample_failure_policy': 'abort_test_no_partial_batch'}}
