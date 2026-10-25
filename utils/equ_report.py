"""Readable terminal reports derived from equilibrium JSONL (stdlib only).

Usage: python -m utils.equ_report PATH [PATH ...]
PATH can be an episode JSONL, an episode directory, or a run archive directory.
Only sibling .txt reports are written; source archives are read-only.
"""
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap


WIDTH = 100


def display(value):
    if value is None:
        return 'not recorded'
    if isinstance(value, float):
        return format(value, '.10g')
    if isinstance(value, dict):
        return '{' + ', '.join(f'{display(key)}: {display(item)}' for key, item in value.items()) + '}'
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(display(item) for item in value) + ']'
    # Escape non-ASCII/control characters instead of sending terminal controls.
    return ''.join(c if c in '\n\t' or 32 <= ord(c) < 127 else ascii(c)[1:-1]
                   for c in str(value))


def option(value):
    if value is None:
        return 'not recorded'
    return '-1 (reject)' if str(value) == '-1' else str(value)


def load_events(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines(keepends=True)
    last = max((i for i, line in enumerate(lines) if line.strip()), default=-1)
    events, warnings = [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except json.JSONDecodeError as exc:
            if index == last and not line.endswith('\n'):
                warnings.append(f'TRUNCATED final JSONL record at line {index + 1}; incomplete record omitted.')
                continue
            raise ValueError(f'{path}: invalid JSON at line {index + 1}: {exc.msg}') from exc
        if not isinstance(event, dict):
            raise ValueError(f'{path}: line {index + 1} is not an event object')
        events.append(event)
    return events, warnings


class TextReport:
    def __init__(self):
        self.lines = []

    def add(self, value='', indent=0):
        wrapper = textwrap.TextWrapper(width=WIDTH, initial_indent=' ' * indent,
                                       subsequent_indent=' ' * (indent + 2),
                                       replace_whitespace=False, drop_whitespace=False,
                                       break_on_hyphens=False)
        for line in display(value).expandtabs(4).split('\n'):
            self.lines.extend(wrapper.wrap(line) or [' ' * indent])

    def heading(self, title, major=False):
        self.add()
        self.add('=' * WIDTH if major else '-' * WIDTH)
        self.add(title)
        self.add('=' * WIDTH if major else '-' * WIDTH)

    def table(self, headers, rows):
        rows = [[display(cell) for cell in row] for row in rows]
        available = WIDTH - 2 * (len(headers) - 1)
        widths = [max([len(header)] + [len(row[i]) for row in rows])
                  for i, header in enumerate(headers)]
        while sum(widths) > available:
            index = max(range(len(widths)), key=widths.__getitem__)
            widths[index] -= 1
        for row in [list(headers), ['-' * width for width in widths]] + rows:
            cells = [textwrap.wrap(cell, width=width, break_on_hyphens=False) or ['']
                     for cell, width in zip(row, widths)]
            for line in range(max(map(len, cells))):
                self.lines.append('  '.join((cell[line] if line < len(cell) else '').ljust(width)
                                            for cell, width in zip(cells, widths)).rstrip())

    def text(self):
        return '\n'.join(self.lines) + '\n'


def first(events, name):
    return next((event for event in events if event.get('event') == name), {})


def sample_counts(events, role):
    requested = sum(event.get('requested_samples', 0) for event in events
                    if event.get('event') == 'query_batch' and event.get('role') == role)
    returned = sum(event.get('status') == 'returned' for event in events
                   if event.get('event') == 'vlm_response' and event.get('role') == role)
    return f'{returned}/{requested}'


def search_warnings(events):
    warnings = []
    if not first(events, 'search_start'):
        warnings.append('INCOMPLETE: search_start is missing.')
    end = first(events, 'search_end')
    if not end:
        warnings.append('INCOMPLETE: search_end is missing; no final decision is assumed.')
    elif end.get('status') not in ('completed', 'early_reject'):
        warnings.append(f'Search ended with status={end.get("status")}.')
    for event in events:
        kind = event.get('event')
        if kind == 'image_input':
            if event.get('expected_dimensions') is False:
                warnings.append(f'UNEXPECTED IMAGE SIZE: candidate {event.get("candidate_id")}, '
                                f'{event.get("dimensions")}; expected [960, 480].')
            if event.get('decode_error'):
                warnings.append(f'IMAGE DECODE ERROR: candidate {event.get("candidate_id")}: '
                                f'{display(event["decode_error"])}')
        elif kind == 'discriminator_candidate_score':
            new = event.get('new_sample_ids', [])
            pool = event.get('sample_pool_ids_used_by_current_code', [])
            if any(sample not in new for sample in pool):
                warnings.append(f'REFERENCE ACCUMULATION: candidate {event.get("candidate_id")} '
                                f'uses {len(pool)} pooled responses, {len(new)} new; '
                                f'denominator={event.get("denominator_used")}. Earlier candidates are included.')
        elif kind == 'discriminator_aggregate' and event.get('values_outside_probability_range'):
            warnings.append('OUT-OF-RANGE Discriminator frequencies are recorded; no correction applied.')
        elif kind == 'policy_snapshot':
            for role, entries in event.get('diagnostics', {}).items():
                for key, checks in entries.items():
                    failed = [name for name in ('finite', 'nonnegative', 'normalized')
                              if checks.get(name) is False]
                    if failed:
                        warnings.append(f'POLICY ANOMALY at update {event.get("update_iteration")}: '
                                        f'{role}[{key}] fails {", ".join(failed)}; '
                                        f'sum={display(checks.get("sum"))}.')
        elif kind == 'vlm_response' and event.get('status') == 'error':
            warnings.append(f'SAMPLE ERROR {event.get("sample_id")}: {display(event.get("error"))}')
        elif kind == 'sample_parse_error':
            warnings.append(f'PARSE ERROR {event.get("sample_id")}: {display(event.get("error"))}')
        elif kind == 'sample_parsed' and event.get('recognized_verdict') is False:
            warnings.append(f'UNRECOGNIZED VERDICT {event.get("sample_id")}: '
                            f'{display(event.get("parsed_verdict"))}; original scoring behavior preserved.')
    return warnings


def render_samples(report, events, role):
    responses = [event for event in events if event.get('event') == 'vlm_response'
                 and event.get('role') == role]
    parsed = {event.get('sample_id'): event for event in events
              if event.get('event') == 'sample_parsed' and event.get('role') == role}
    report.add(f'{role.upper()} INDIVIDUAL OPINIONS')
    if not responses:
        report.add('No sample returns/errors recorded. See query batches and search status.', 2)
        return
    rows = []
    for response in responses:
        value = parsed.get(response.get('sample_id'), {})
        interpretation = (option(value['parsed_object_id']) if 'parsed_object_id' in value
                          else value.get('parsed_verdict'))
        rows.append([response.get('sample_id'),
                     option(response.get('candidate_id')) if role == 'discriminator' else 'all inputs',
                     response.get('completion_order'), interpretation, response.get('status')])
    report.table(['Sample', 'Evaluated option', 'Completion', 'Parsed result', 'Status'], rows)
    report.add('Raw returned text (display-wrapped; exact whitespace is retained in JSONL):')
    for response in responses:
        report.add(f'{response.get("sample_id")}:', 2)
        if 'raw_content' in response:
            report.add(response['raw_content'], 4)
        else:
            report.add(f'ERROR: {display(response.get("error"))}', 4)


def render_search(report, sid, events, warnings):
    start = first(events, 'search_start')
    end = first(events, 'search_end')
    report.heading(f'SEARCH {sid} | agent step {display(start.get("agent_step"))}', major=True)
    report.add(f'Status: {end.get("status", "INCOMPLETE")} | '
               f'Selected option: {option(end.get("selected_candidate_id"))} | '
               f'Duration: {display(end.get("duration_seconds"))} s')
    report.add('Result describes this search, not navigation success.')
    for warning in warnings:
        report.add('WARNING: ' + warning)
    parameters = first(events, 'search_parameters')
    report.heading('1. INPUTS AND SETTINGS')
    report.add(f'Input candidate order: {display(start.get("candidate_order"))}')
    if parameters:
        report.add('Recorded search parameters: ' + display({k: parameters[k] for k in (
            'iterations', 'generator_samples', 'discriminator_samples', 'bias',
            'lambda_G', 'lambda_D', 'eta_G', 'eta_D') if k in parameters}))
    report.add(f'Logical samples returned/requested: G={sample_counts(events, "generator")}, '
               f'D={sample_counts(events, "discriminator")}. HTTP retries/token usage are not counted.')
    images = [event for event in events if event.get('event') == 'image_input']
    for image in images:
        dimensions = 'x'.join(map(str, image['dimensions'])) if 'dimensions' in image else 'not decoded'
        report.add(f'Input {image.get("input_position")}: candidate {image.get("candidate_id")}, '
                   f'{dimensions}, {display(image.get("byte_count"))} bytes', 2)
        report.add('Archive (relative to run root): ' + display(image.get('archive_path')), 4)
        report.add('Original source: ' + display(image.get('source_path')), 4)
        report.add('SHA-256: ' + display(image.get('sha256')), 4)
        if image.get('render_completion'):
            report.add('Completed render request: ' + display(image['render_completion'].get('request_id')), 4)
            report.add('First-view SHA-256: ' + display(image['render_completion'].get('first_view_sha256')), 4)
    if not images:
        report.add('No archived input metadata recorded.')

    report.heading('2. GENERATOR RESPONSES AND AGGREGATION')
    render_samples(report, events, 'generator')
    gen = first(events, 'generator_aggregate')
    if gen:
        report.add('Generator frequencies (sample frequencies, not API confidence scores):')
        report.add('Actual denominator: ' + display(gen.get('denominator_used')))
        frequencies = dict(gen.get('p_vlm_all_candidates', {}))
        for key, value in gen.get('p_vlm_observed', {}).items():
            frequencies.setdefault(key, value)
        supported = {str(cid) for cid in gen.get('policy_candidate_order', [])}
        report.table(['Option', 'Votes', 'Frequency', 'In policy'], [
            [option(key), gen.get('counts', {}).get(key, 0), value,
             'yes' if str(key) in supported else 'no']
            for key, value in frequencies.items()])
        report.add('Policy candidate order: ' + display(gen.get('policy_candidate_order')))
        report.add('P_G correct:   ' + display(gen.get('p_vlm_correct')))
        report.add('P_G incorrect: ' + display(gen.get('p_vlm_incorrect')))
        report.add('The incorrect row is computed as 1 - correct, not separately queried.')
        excluded = [key for key in gen.get('p_vlm_all_candidates', {}) if key not in supported]
        if excluded:
            report.add('Excluded from equilibrium support (zero Generator votes): ' + ', '.join(excluded))
    else:
        report.add('Generator aggregate not recorded.')

    report.heading('3. DISCRIMINATOR RESPONSES AND ACTUAL SCORING POOLS')
    render_samples(report, events, 'discriminator')
    disc = first(events, 'discriminator_aggregate')
    scores = [event for event in events if event.get('event') == 'discriminator_candidate_score']
    for score in scores:
        report.add(f'Candidate {score.get("candidate_id")} (internal key {score.get("internal_key")}):')
        report.add('New samples: ' + display(score.get('new_sample_ids')), 2)
        report.add('Actual scoring pool: ' + display(score.get('sample_pool_ids_used_by_current_code')), 2)
        report.add(f'Correct count used: {score.get("correct_count_used")} / '
                   f'denominator {score.get("denominator_used")}', 2)
    if disc:
        reject = disc.get('reject_all', {})
        report.add('Reject-all source: ' + display(reject.get('source')))
        if reject.get('source') == 'derived_complement':
            report.add('Derived from the sole discriminated candidate; no additional VLM queries.')
        elif reject.get('source') == 'queried_all_input_images':
            report.add('Queried on all input images; incorrect responses count toward rejection.')
            report.add('Reject-all sample IDs: ' + display(reject.get('sample_ids')))
        report.add(f'Reject count: {display(reject.get("count"))} / '
                   f'denominator {display(reject.get("denominator_used"))}')
    else:
        report.add('No Discriminator aggregate recorded; consult search status.')

    report.heading('4. BIAS AND NORMALIZATION INPUTS')
    layout = first(events, 'policy_layout')
    mapping = layout.get('discriminator_key_to_candidate_id', disc.get('internal_key_to_candidate_id', {}))
    order = layout.get('policy_candidate_order', gen.get('policy_candidate_order', []))
    report.add('Policy option order: ' + display(order))
    report.add('Discriminator internal key -> candidate ID: ' + display(mapping))
    if layout:
        report.add('Generator normalization input (bias already added):')
        for verdict, values in layout.get('generator_normalization_input', {}).items():
            report.add(f'{verdict}: {display(values)}', 2)
    if disc:
        report.add('Discriminator columns below are [correct, incorrect].')
        report.add('Recorded bias: ' + display(disc.get('bias')))
        report.table(['Internal key', 'Option', 'Before bias', 'Normalization input'], [
            [key, option(mapping.get(str(key))), values, disc.get('normalization_input', {}).get(key)]
            for key, values in disc.get('p_vlm_before_bias', {}).items()])
    if not layout:
        report.add('Normalization inputs for initialized policies were not recorded.')

    report.heading('5. COMPLETE POLICY SNAPSHOTS')
    report.add('G columns: weights over options conditional on the verdict.')
    report.add('D columns: weights over verdicts conditional on the option.')
    report.add('These are policy weights, not calibrated real-world confidence.')
    snapshots = [event for event in events if event.get('event') == 'policy_snapshot']
    if not snapshots:
        if end.get('status') == 'early_reject':
            report.add('Early rejection: policies were not initialized; no updates/checkpoints expected.')
        else:
            report.add('No snapshots recorded; do not assume initialization or updates completed.')
    else:
        reverse = {str(cid): str(key) for key, cid in mapping.items()}
        rows = []
        for snapshot in snapshots:
            iteration = snapshot.get('update_iteration')
            label = 'initial(0)' if iteration == 0 else f'{iteration} ({display(snapshot.get("progress_percent"))}%)'
            G, D = snapshot.get('pi_G', {}), snapshot.get('pi_D', {})
            count = max(len(order), *(len(values) for values in G.values()), 0)
            for index in range(count):
                cid = order[index] if index < len(order) else None
                key = reverse.get(str(cid)) if cid is not None else None

                def entry(values, position):
                    return values[position] if values is not None and position < len(values) else None

                rows.append([label, option(cid) if cid is not None else f'index {index} (unmapped)',
                             entry(G.get('correct'), index), entry(G.get('incorrect'), index),
                             entry(D.get(key), 0), entry(D.get(key), 1)])
        report.table(['Update', 'Option', 'G(correct)', 'G(incorrect)', 'D(correct)', 'D(incorrect)'], rows)
        failures = [warning for warning in warnings if warning.startswith('POLICY ANOMALY')]
        if failures:
            report.add('Recorded numerical checks failed; values are displayed without repair.')
        else:
            diagnostics = [checks for snapshot in snapshots for entries in snapshot.get('diagnostics', {}).values()
                           for checks in entries.values()]
            if diagnostics and all(all(checks.get(name) is True for name in ('finite', 'nonnegative', 'normalized'))
                                   for checks in diagnostics):
                report.add('Recorded checks: all supplied policy diagnostics are finite/nonnegative/normalized.')
            else:
                report.add('Complete numerical diagnostics were not recorded.')

    report.heading('6. DECISION')
    if end:
        report.add(f'Status: {end.get("status")}')
        report.add('Selected option: ' + option(end.get('selected_candidate_id')))
        report.add('Selected policy index: ' + display(end.get('selected_policy_index')))
        report.add('Equilibrium updates completed: ' + display(end.get('equilibrium_updates_completed')))
        if end.get('reason'):
            report.add('Reason: ' + display(end['reason']))
        if end.get('error'):
            report.add('Error: ' + display(end['error']))
    else:
        report.add('INCOMPLETE: no search_end. Final choice and completed update count are unknown.')


def episode_report(path):
    path = Path(path)
    events, warnings = load_events(path)
    manifest_path = next((parent / 'run.json' for parent in path.parents
                          if (parent / 'run.json').is_file()), None)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path else {}
    if not manifest_path:
        warnings.append('Run manifest unavailable; model/configuration may be unknown.')
    start, end = first(events, 'episode_start'), first(events, 'episode_end')
    if not start:
        warnings.append('INCOMPLETE: episode_start is missing.')
    if not end:
        warnings.append('INCOMPLETE: episode_end is missing; episode outcome is unknown.')
    elif end.get('status') != 'completed':
        warnings.append(f'Episode ended with status={end.get("status")}.')
    searches = OrderedDict()
    for event in events:
        if event.get('search_id') is not None:
            searches.setdefault(event['search_id'], []).append(event)
    per_search_warnings = {sid: search_warnings(search) for sid, search in searches.items()}
    if end and end.get('search_invocations') != len(searches):
        warnings.append('Recorded search count differs from the number of search IDs present.')
    report = TextReport()
    report.heading(f'EPISODE {start.get("episode_number", "?")} - EQUILIBRIUM SEARCH REPORT', major=True)
    report.add('Run: ' + display(start.get('run_id', manifest.get('run_id'))))
    report.add('JSONL source: ' + str(path))
    report.add('Manifest: ' + display(manifest_path))
    report.add('Status: ' + end.get('status', 'INCOMPLETE'))
    report.add('Dataset episode ID: ' + display(start.get('dataset_episode_id')))
    report.add('Scene: ' + display(start.get('scene_id')))
    report.add('Instruction: ' + display(start.get('instruction')))
    parsed = first(events, 'instruction_parsed')
    if parsed:
        report.add('Parsed instruction: ' + display(parsed.get('parsed_instruction')))
    api = manifest.get('api_configuration', {})
    report.add('API base: ' + display(api.get('api_base')))
    for role, request in api.get('requests', {}).items():
        report.add(f'{role.capitalize()} model/settings: {display(request)}')
    report.add('Display: numbers rounded to 10 significant digits; small values use scientific notation.')
    report.add('Long text wraps at 100 columns; non-ASCII/control characters are escaped. '
               'JSONL remains the exact, full-precision source. No probabilities are recalculated.')
    report.add('Logical samples are not billed HTTP request counts; internal retries/usage are not captured.')
    report.add('Manifest aggregation label: ' + display(manifest.get('discriminator_aggregation')))
    report.add('Candidate image protocol: ' + display(manifest.get('candidate_image_protocol')))
    report.heading('EPISODE OUTCOME')
    report.add('Environment steps: ' + display(end.get('steps')))
    report.add('Recorded metrics: ' + display(end.get('metrics')))
    report.add(f'Search IDs present: {len(searches)}; recorded search invocations: '
               f'{display(end.get("search_invocations"))}; recorded equilibrium runs: '
               f'{display(end.get("equilibrium_runs"))}')
    if end.get('error'):
        report.add('Episode error: ' + display(end['error']))
    report.heading('WARNINGS')
    if not warnings and not any(per_search_warnings.values()):
        report.add('No warnings in the recorded events. This does not verify image freshness or correctness.')
    for warning in warnings:
        report.add('WARNING: ' + warning)
    for sid, items in per_search_warnings.items():
        for warning in items:
            report.add(f'WARNING [search {sid}]: {warning}')
    report.heading('SEARCH OVERVIEW')
    if searches:
        report.table(['Search', 'Step', 'Inputs', 'G ret/req', 'D ret/req', 'Status', 'Selected', 'Warnings'], [
            [sid, first(search, 'search_start').get('agent_step'),
             len([e for e in search if e.get('event') == 'image_input']),
             sample_counts(search, 'generator'), sample_counts(search, 'discriminator'),
             first(search, 'search_end').get('status', 'INCOMPLETE'),
             option(first(search, 'search_end').get('selected_candidate_id')),
             len(per_search_warnings[sid])] for sid, search in searches.items()])
    else:
        report.add('No searches recorded.')
    parsing = first(events, 'parsing_response')
    if parsing:
        report.heading('RAW INSTRUCTION-PARSING RESPONSE')
        report.add(parsing.get('raw_content'), 2)
    for sid, search in searches.items():
        render_search(report, sid, search, per_search_warnings[sid])
    return report.text()


def write_episode_report(path):
    path = Path(path)
    text = episode_report(path)
    output = path.with_suffix('.txt')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='ascii', dir=output.parent,
                                         prefix='.equilibrium-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args(argv)
    paths = set()
    for path in args.paths:
        if path.is_file() and path.suffix == '.jsonl':
            paths.add(path.resolve())
        elif path.is_dir():
            paths.update(item.resolve() for item in path.rglob('equilibrium.jsonl'))
        else:
            parser.error(f'Not an archive directory or JSONL file: {path}')
    if not paths:
        parser.error('No equilibrium.jsonl files found')
    failed = False
    for path in sorted(paths):
        try:
            print(write_episode_report(path))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f'Could not generate report for {path}: {exc}', file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
