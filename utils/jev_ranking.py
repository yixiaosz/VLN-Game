"""Jev-native full-distribution methods; existing chat ranking stays separate."""
import concurrent.futures
from functools import partial
import json
import threading

import numpy as np

from utils.chat_utils import APIRequestError
from utils.instruction_lookup import InstructionLookup
from utils.jev_client import build_request, request_decision

SOLVER = 'jev_full_support_arrays_vlngame_accumulated_500_v1'


def row_softmax(logits):
    exponents = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponents / exponents.sum(axis=1, keepdims=True)


def solve_equilibrium(g, d, checkpoint=None):
    """VLN-Game equations with explicit axes; no sampled-support filtering."""
    g, d = np.asarray(g, dtype=np.float64), np.asarray(d, dtype=np.float64)
    if (g.ndim != 1 or g.shape != d.shape or len(g) < 2 or not np.isfinite(g).all()
            or not np.isfinite(d).all() or np.any(g < 0) or np.any(g > 1)
            or np.any(d < 0) or np.any(d > 1) or not np.isclose(g.sum(), 1, atol=1e-6, rtol=0)):
        raise ValueError('Invalid full-support Generator/Discriminator vectors')
    G = np.stack([g, 1-g]) + 0.05
    D = np.column_stack([d, 1-d]) + 0.05
    G0, D0 = G/G.sum(axis=1, keepdims=True), D/D.sum(axis=1, keepdims=True)
    G, D = G0.copy(), D0.copy()
    QG, QD = np.zeros_like(G), np.zeros_like(D)
    if checkpoint:
        checkpoint(0, G, D, QG, QD)
    for t in range(1, 501):
        QG += D.T / (2*t)
        QD += G.T / (2*t)
        # Both accumulated payoffs use previous policies before either is replaced.
        G = row_softmax((QG + 0.1*np.log(G0)) / (0.1 + 1/(0.1*t)))
        D = row_softmax((QD + 0.1*np.log(D0)) / (0.1 + 1/(0.1*t)))
        if checkpoint and t in (125, 250, 375, 500):
            checkpoint(t, G, D, QG, QD)
    return G, D


class JevEquilibriumRanking:
    def __init__(self, text_queries, logger=None, episode_trace=None, lookup=None):
        self.text_description = text_queries
        self.lookup = lookup if lookup is not None else InstructionLookup()
        parsed = self.lookup.get(text_queries)
        self.response_message = repr(parsed)  # Existing agent consumes literal_eval, including apostrophes.
        if episode_trace is not None:
            episode_trace.emit('parsing_response', raw_content=self.response_message,
                               source='frozen_instruction_lookup', table_sha256=self.lookup.hash,
                               instruction=text_queries)

    def _batch(self, payload, samples, role, prefix, trace, **context):
        cancel = threading.Event()
        if trace:
            trace.emit('query_batch', role=role, requested_samples=samples, **context)
        results, errors = {}, []
        def invoke(n):
            observer = partial(trace.emit, sample_id=f'{prefix}-{n}', role=role, **context) if trace else None
            return request_decision(payload, role, observer, cancel)
        with concurrent.futures.ThreadPoolExecutor(max_workers=samples) as executor:
            futures = {executor.submit(invoke, n): n for n in range(1, samples+1)}
            try:
                for order, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    n = futures[future]
                    try:
                        data, normalized = future.result()
                        results[n] = normalized
                    except Exception as exc:
                        errors.append(exc)
                        cancel.set()
                        if trace:
                            trace.emit('vlm_response', role=role, sample_id=f'{prefix}-{n}', status='error',
                                       completion_order=order, error={'type': type(exc).__name__, 'message': str(exc)}, **context)
                    else:
                        if trace:
                            trace.emit('vlm_response', role=role, sample_id=f'{prefix}-{n}', status='returned',
                                       completion_order=order, raw_content=json.dumps(data), **context)
                            trace.emit('sample_parsed', role=role, sample_id=f'{prefix}-{n}',
                                       native_probabilities=normalized, **context)
            finally:
                cancel.set()
        if errors or len(results) != samples:
            raise APIRequestError(f'Jev {role} batch incomplete; no partial aggregation')
        # Fixed sample order makes summation independent of completion order.
        return [results[n] for n in range(1, samples+1)]

    def _search(self, images, ids, samples, game, trace):
        options = list(ids) + [-1]
        if trace:
            trace.iterations = 500 if game else 0
            trace.emit('search_parameters', backend='jev', method='game' if game else 'generator' if samples == 1 else 'ranking',
                       aggregation='equal_weight_probability_mean_v1', iterations=500 if game else 0,
                       generator_samples=samples, discriminator_samples=5 if game else 0,
                       bias=0.05 if game else None, lambda_G=0.1, lambda_D=0.1, eta_G=0.1, eta_D=0.1,
                       tie_break='first_candidate_order_reject_last', solver=SOLVER if game else 'not_applicable')
        try:
            payload = build_request(self.text_description, ids, images, 'choice')
            replies = self._batch(payload, samples, 'generator', 'g', trace, input_positions=list(range(len(ids))))
            g = np.mean([[reply['probabilities'][str(cid)] for cid in options] for reply in replies], axis=0)
            if trace:
                trace.emit('generator_aggregate', aggregation='equal_weight_probability_mean_v1', denominator_used=samples,
                           policy_candidate_order=options, p_vlm_all_candidates=dict(zip(map(str, options), g)),
                           p_vlm_correct=g, p_vlm_incorrect=1-g)
            if game:
                d = []
                for pos, cid in enumerate(ids):
                    payload = build_request(self.text_description, [cid], [images[pos]], 'match')
                    replies = self._batch(payload, 5, 'discriminator', f'd-{cid}', trace,
                                          candidate_id=cid, input_positions=[pos])
                    d.append(float(np.mean([reply['noul'] for reply in replies])))
                if len(ids) == 1:
                    rejection = 1-d[0]
                    source = 'derived_complement'
                else:
                    payload = build_request(self.text_description, ids, images, 'reject')
                    replies = self._batch(payload, 5, 'discriminator', 'd-reject', trace,
                                          candidate_id=-1, input_positions=list(range(len(ids))))
                    rejection = float(np.mean([reply['noul'] for reply in replies]))
                    source = 'queried_all_input_images'
                d = np.asarray(d + [rejection])
                if trace:
                    trace.emit('discriminator_aggregate', aggregation='candidate_local_noul_mean_v1',
                               policy_candidate_order=options, match_probabilities=d, verdict_probabilities=np.column_stack([d, 1-d]),
                               reject_all={'source': source, 'probability': rejection}, denominator_used=5)
                    trace.emit('policy_layout', solver=SOLVER, policy_candidate_order=options,
                               verdict_order=['correct', 'incorrect'], generator_axes=['verdict', 'candidate'],
                               discriminator_axes=['candidate', 'verdict'],
                               generator_normalization_input=np.stack([g, 1-g])+0.05,
                               discriminator_normalization_input=np.column_stack([d, 1-d])+0.05)
                def checkpoint(t, G, D, QG, QD):
                    if trace:
                        trace.updates_completed = t
                        trace.snapshot(t, G, D)
                        trace.emit('payoff_snapshot', update_iteration=t, Q_G=QG, Q_D=QD)
                G, _ = solve_equilibrium(g, d, checkpoint)
                scores = G[0]
            else:
                scores = g
            index = int(np.argmax(scores))
            selected = options[index]
            if trace:
                trace.emit('generator_selection', selected_candidate_id=selected, option_order=options,
                           tied_candidate_ids=[cid for cid, score in zip(options, scores) if score == scores[index]],
                           tie_break='first_candidate_order_reject_last')
                trace.finish('completed', selected_candidate_id=selected,
                             selected_policy_index=index if game else None, reason='jev_policy_argmax' if game else 'jev_probability_argmax')
            return selected
        except Exception as exc:
            if trace:
                trace.finish('error', error={'type': type(exc).__name__, 'message': str(exc)})
            raise

    def generator_search(self, base64_image_list, candidate_id, num_samples=1, trace=None):
        return self._search(base64_image_list, candidate_id, num_samples, False, trace)

    def equilibrium_search(self, base64_image_list, candidate_id, trace=None):
        return self._search(base64_image_list, candidate_id, 5, True, trace)
