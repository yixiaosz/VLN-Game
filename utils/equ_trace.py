"""Append-only equilibrium traces; no navigation or probability calculations."""
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
from utils.equ_report import write_episode_report


def json_value(value):
    """Keep nonfinite values explicit while producing standards-compliant JSON."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, Path):
        return str(value)
    return value


def policy_diagnostics(policy):
    result = {}
    for key, values in policy.items():
        values = np.asarray(values)
        finite = bool(np.isfinite(values).all())
        result[str(key)] = {
            'finite': finite,
            'nonnegative': bool((values >= 0).all()),
            'sum': np.sum(values),
            'normalized': finite and bool(np.isclose(np.sum(values), 1.0)),
        }
    return result


def source_identity():
    root = Path(__file__).resolve().parents[1]
    identity = {'sha256': {}}
    for name in ('utils/equ_ranking.py', 'utils/equ_trace.py', 'utils/equ_report.py', 'utils/chat_utils.py',
                 'agents/vlnav_agent_vlm.py', 'agents/system_prompt.py',
                 'main_vln.py', 'utils/vis_gui.py', 'utils/vis.py', 'utils/candidate_images.py', 'configs/mapping_base.yaml',
                 'configs/vlobjectnav_hm3d.yaml', 'arguments.py'):
        identity['sha256'][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    try:
        identity['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
        identity['git_dirty'] = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        identity['git_commit'] = None
    return identity


class EquilibriumTraceRun:
    def __init__(self, log_dir, metadata, prompts, logger):
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + f'-{os.getpid()}'
        self.root = Path(log_dir) / 'traces' / run_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.run_id = run_id
        self.logger = logger
        manifest = dict(metadata, schema_version=1, run_id=run_id, prompts=prompts,
                        source=source_identity(),
                        discriminator_aggregation='candidate_local_responses_v1',
                        nonfinite_encoding='NaN, Infinity, -Infinity are JSON strings',
                        response_scope='Final logical sample returns/errors; internal HTTP retries are not individually recorded')
        with (self.root / 'run.json').open('x') as stream:
            json.dump(json_value(manifest), stream, indent=2, allow_nan=False)
        logger.info('Equilibrium trace run: %s', self.root)

    def start_episode(self, number, dataset_episode_id, scene_id, instruction):
        return EpisodeTrace(self, number, dataset_episode_id, scene_id, instruction)


class EpisodeTrace:
    def __init__(self, run, number, dataset_episode_id, scene_id, instruction):
        self.run = run
        self.number = number
        self.directory = run.root / 'episodes' / f'{number:03d}'
        self.directory.mkdir(parents=True)
        self.path = self.directory / 'equilibrium.jsonl'
        self.stream = self.path.open('x')
        self.search_count = 0
        self.equilibrium_runs = 0
        self.current_search = None
        self.emit('episode_start', dataset_episode_id=dataset_episode_id,
                  scene_id=scene_id, instruction=instruction)
        run.logger.info('Episode %d trace: %s', number, self.path)

    def emit(self, event, **fields):
        record = dict(event=event, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      run_id=self.run.run_id, episode_number=self.number, **fields)
        self.stream.write(json.dumps(json_value(record), allow_nan=False) + '\n')
        self.stream.flush()

    def start_search(self, agent_step, candidate_ids):
        self.search_count += 1
        self.current_search = SearchTrace(self, self.search_count, agent_step, candidate_ids)
        return self.current_search

    def finish(self, status, steps, metrics=None, error=None):
        if self.stream.closed:
            return
        if self.current_search and not self.current_search.ended:
            self.current_search.finish(status, error=error)
        self.emit('episode_end', status=status, steps=steps, metrics=metrics,
                  error=error, search_invocations=self.search_count,
                  equilibrium_runs=self.equilibrium_runs)
        self.stream.close()
        try:
            report = write_episode_report(self.path)
            self.run.logger.info('Episode %d text report: %s', self.number, report)
        except Exception:
            # Presentation failures must not alter navigation or episode metrics.
            self.run.logger.exception('Could not generate text report for %s; JSONL retained', self.path)


class SearchTrace:
    def __init__(self, episode, number, agent_step, candidate_ids):
        self.episode = episode
        self.number = number
        self.started = time.monotonic()
        self.ended = False
        self.initialized = False
        self.updates_completed = 0
        self.iterations = 500
        self.images = []
        self.emit('search_start', agent_step=agent_step, candidate_order=candidate_ids,
                  generator_prompt_ref='run.json#/prompts/generator',
                  discriminator_prompt_ref='run.json#/prompts/discriminator')
        episode.run.logger.info('Episode %d search %d at agent step %d; trace=%s',
                                episode.number, number, agent_step, episode.path)

    def emit(self, event, **fields):
        self.episode.emit(event, search_id=self.number, **fields)

    def archive_image(self, candidate_id, input_position, source_path, data, render_completion=None):
        directory = self.episode.directory / 'images'
        directory.mkdir(exist_ok=True)
        path = directory / f'search-{self.number:03d}-input-{input_position:03d}-candidate-{candidate_id}.jpeg'
        with path.open('xb') as stream:
            stream.write(data)
        image = dict(candidate_id=candidate_id, input_position=input_position,
                     source_path=str(source_path),
                     archive_path=str(path.relative_to(self.episode.run.root)),
                     sha256=hashlib.sha256(data).hexdigest(), byte_count=len(data))
        if render_completion is not None:
            image['render_completion'] = render_completion
        try:
            with Image.open(BytesIO(data)) as decoded:
                image['dimensions'] = list(decoded.size)
                image['format'] = decoded.format
                image['expected_dimensions'] = decoded.size == (960, 480)
        except Exception as exc:
            image['decode_error'] = {'type': type(exc).__name__, 'message': str(exc)}
        self.images.append(image)
        self.emit('image_input', **image)

    def snapshot(self, iteration, pi_G, pi_D):
        if iteration == 0:
            self.initialized = True
            self.episode.equilibrium_runs += 1
        self.emit('policy_snapshot', phase='initial' if iteration == 0 else 'updated',
                  update_iteration=iteration, progress_percent=iteration / self.iterations * 100,
                  pi_G=pi_G, pi_D=pi_D,
                  diagnostics={'pi_G': policy_diagnostics(pi_G),
                               'pi_D': policy_diagnostics(pi_D)})

    def finish(self, status, selected_candidate_id=None, selected_policy_index=None,
               reason=None, error=None):
        if self.ended:
            return
        self.emit('search_end', status=status, reason=reason, error=error,
                  selected_candidate_id=selected_candidate_id,
                  selected_policy_index=selected_policy_index,
                  policies_initialized=self.initialized,
                  equilibrium_updates_completed=self.updates_completed,
                  duration_seconds=time.monotonic() - self.started)
        self.ended = True
        self.episode.run.logger.info('Episode %d search %d: %s, selected=%s, updates=%d',
                                    self.episode.number, self.number, status,
                                    selected_candidate_id, self.updates_completed)
