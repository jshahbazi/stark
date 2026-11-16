"""Offline aggregate analysis for the pinned InfiniteBench performance case."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ORDER = ('before', 'candidate', 'candidate', 'before')
PAIRS = ((1, 2), (4, 3))
INPUT = 88398
CONTEXT = 98304
WORK = ('rounds', 'proposed', 'accepted', 'verified_rows')
METRICS = ('prefill_seconds', 'prefill_tokens_per_second', 'decode_seconds',
           'useful_decode_tokens_per_second', 'wall_seconds')
RATE_METRICS = {'prefill_tokens_per_second', 'useful_decode_tokens_per_second'}


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sha(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] * (upper - position) + values[upper] * (position - lower) if lower != upper else values[lower]


def distribution(values):
    values = [value for value in values if finite(value)]
    if not values:
        return {key: 0 if key == 'n' else None for key in ('n', 'min', 'median', 'max', 'q1', 'q3', 'iqr', 'mad')}
    median = statistics.median(values)
    q1, q3 = quantile(values, 0.25), quantile(values, 0.75)
    return {'n': len(values), 'min': min(values), 'median': median, 'max': max(values),
            'q1': q1, 'q3': q3, 'iqr': q3 - q1,
            'mad': statistics.median(abs(value - median) for value in values)}


def change_summary(values):
    values = [value for value in values if finite(value)]
    return {**distribution(values), 'candidate_wins': sum(value > 0 for value in values),
            'candidate_losses': sum(value < 0 for value in values), 'ties': sum(value == 0 for value in values)}


def metrics(row):
    result = {key: row.get(key) for key in METRICS}
    prefill = row.get('prefill_seconds')
    result['prefill_tokens_per_second'] = INPUT / prefill if finite(prefill) and prefill > 0 else None
    emitted, decode = row.get('output_tokens'), row.get('decode_seconds')
    result['useful_decode_tokens_per_second'] = ((emitted - 1) / decode
        if type(emitted) is int and emitted > 1 and finite(decode) and decode > 0 else None)
    return result


def signature(row):
    return (row.get('output_token_ids_sha256'), row.get('text_sha256'), row.get('input_tokens'),
            row.get('output_tokens'), row.get('sampled_tokens'), row.get('finish_reason'), row.get('natural_eos'),
            *(row.get('mtp_counters', {}).get(key) for key in WORK))


def source_checks(manifest):
    model = Path('/Users/johnshahbazian/Library/Application Support/Stark/models/qwen38-flash-next-lily-q4-xl-0f94842e')
    checks = {}
    bindings = {
        'before_worker': (Path('/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-baseline-target/release/stark-generate'), manifest.get('worker_sha256', {}).get('before')),
        'candidate_worker': (ROOT / 'runtime/lily-verifier-study/candidate-source/target/release/stark-generate', manifest.get('worker_sha256', {}).get('candidate')),
        'candidate_patch': (ROOT / 'docs/archive/experiments/2026-11-15-lily-verifier/projection-reuse-v2-as-run.patch', manifest.get('candidate_patch_sha256')),
        'harness': (HERE / 'run.py', manifest.get('harness_sha256')),
        'preparer': (HERE / 'prepare.py', manifest.get('prepare_sha256')),
        'generation_helper': (ROOT / 'runtime/lily-slack-long/run.py', manifest.get('generation_helper_sha256')),
        'identity_helper': (ROOT / 'runtime/lily-topk-policy/run_performance.py', manifest.get('model_identity_helper_sha256')),
        'model_identity_receipt': (Path('/Users/johnshahbazian/development/stark-llama-v1-study/runtime/llama-v1-lily-baseline-20261115-v2/identity.json'), manifest.get('model_identity_receipt_sha256')),
        'native_source_receipt': (model / 'native-source.json', manifest.get('native_source_sha256')),
        'model_config': (model / 'config.json', manifest.get('config_sha256')),
        'tokenizer': (model / 'tokenizer.json', manifest.get('source', {}).get('tokenizer_json_sha256')),
        'template_helper': (ROOT / 'runtime/build-candidate/release/stark-template', manifest.get('source', {}).get('template_helper_sha256')),
    }
    for name, (path, expected) in bindings.items():
        actual = sha(path)
        checks[name] = {'matched': isinstance(expected, str) and actual == expected,
                        'expected_sha256': expected, 'current_sha256': actual}
    checks['preparation_record'] = {'matched': manifest.get('source') == read(HERE / 'preparation.json')}
    return checks


def host_summary(rows):
    hosts = [row.get('host') for row in rows if isinstance(row.get('host'), dict)]
    result = {'requests': len(rows), 'host_records': len(hosts)}
    for key in ('samples', 'external_cpu_max', 'external_model_cpu_max', 'thermal_max', 'sampling_failures'):
        result[key] = distribution([host.get(key) for host in hosts])
    for key in ('compiler_active', 'sampling_gap', 'eligible'):
        result[key] = {label: sum(host.get(key) is value for host in hosts)
                       for label, value in (('true', True), ('false', False), ('missing', None))}
    result['screen_failure_records'] = sum('screen_failure_type' in host for host in hosts)
    result['pageouts_total'] = sum((host.get('paging_delta') or {}).get('pageouts', 0) for host in hosts)
    result['swapouts_total'] = sum((host.get('paging_delta') or {}).get('swapouts', 0) for host in hosts)
    return result


def request_counts(rows):
    return {'requests': len(rows), 'output_tokens': distribution([row.get('output_tokens') for row in rows]),
            'sampled_tokens': distribution([row.get('sampled_tokens') for row in rows]),
            'natural_eos': sum(row.get('natural_eos') is True for row in rows),
            'length_finishes': sum(row.get('finish_reason') == 'length' for row in rows),
            'progress_events': distribution([row.get('progress_events') for row in rows]),
            'mtp_counters': {key: distribution([row.get('mtp_counters', {}).get(key) for row in rows]) for key in WORK}}


def analyze(directory):
    manifest = read(directory / 'manifest.json')
    if not isinstance(manifest, dict):
        return {'schema': 'stark-lily-long-context-analysis-v1', 'status': 'incomplete', 'missing_files': ['manifest.json']}
    sessions, missing = {}, []
    for index in range(1, 5):
        value = read(directory / f'session-{index}.json')
        if isinstance(value, dict):
            sessions[index] = value
        else:
            missing.append(f'session-{index}.json')
    sources = source_checks(manifest)
    integrity = {
        'manifest_completed': manifest.get('completed') is True,
        'source_hashes_match': all(value['matched'] for value in sources.values()),
        'fixed_configuration': (manifest.get('session_order') == list(ORDER)
            and manifest.get('source', {}).get('input_tokens') == INPUT
            and manifest.get('context_tokens') == CONTEXT and manifest.get('max_new_tokens') == 4096
            and manifest.get('warmup_max_new_tokens') == 64 and manifest.get('mtp_drafts') == 2
            and manifest.get('prefix_cache') is False and manifest.get('vision_loaded') is True
            and manifest.get('repetitions_per_session') == 4 and manifest.get('expected_measured_requests') == 16
            and manifest.get('expected_warmups') == 4 and manifest.get('stream') is True
            and manifest.get('seed') == 42 and manifest.get('sampling') == {
                'mode': 'stochastic', 'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}),
        'all_sessions_present': len(sessions) == 4,
    }
    errors, flat, grids, guards = [], [], {}, {}
    rows_by_session = {}
    for index, session in sessions.items():
        rows = session.get('requests', [])
        rows_by_session[index] = {row.get('repetition'): row for row in rows}
        reps = [row.get('repetition') for row in rows]
        grids[str(index)] = {'requests': len(rows), 'complete': len(rows) == 5 and set(reps) == set(range(5))}
        guards[str(index)] = session.get('model_guard_passed')
        if session.get('index') != index or session.get('arm') != ORDER[index - 1]:
            errors.append({'session': index, 'check': 'session_identity'})
        if session.get('failure_type') is not None:
            errors.append({'session': index, 'check': 'session_failure'})
        if session.get('input_tokenization_match') is not True or session.get('preflight_plan_ok') is not True:
            errors.append({'session': index, 'check': 'tokenizer_or_plan_probe'})
        ready = session.get('ready', {})
        if (ready.get('mtp_recommended_drafts') != 2 or ready.get('max_context_tokens', 0) < CONTEXT
                or ready.get('image_family_profile') != 'qwen38-lily-retained-bf16-vision-v1'):
            errors.append({'session': index, 'check': 'readiness'})
        for row in rows:
            flat.append((index, ORDER[index - 1], row))
            checks = []
            rep = row.get('repetition')
            warmup = rep == 0
            if rep not in range(5) or row.get('warmup') is not warmup:
                checks.append('repetition_or_warmup')
            if row.get('request_status') != 'ok' or row.get('transport_failure_type') is not None:
                checks.append('request_status')
            if row.get('input_tokens') != INPUT or row.get('actual_drafts') != 2 or row.get('mtp_verification') != 'batched-decode-exact-v1':
                checks.append('input_or_route')
            if row.get('prefix_reused_tokens') != 0:
                checks.append('prefix_reuse')
            if not all(row.get(key) is True for key in ('counts_valid', 'automatic_depth_valid', 'timings_valid', 'prefix_reuse_valid', 'same_work_and_output')):
                checks.append('native_validity_flags')
            for metric, value in metrics(row).items():
                if not finite(value) or value <= 0:
                    checks.append('finite_timing_' + metric)
            emitted, sampled = row.get('output_tokens'), row.get('sampled_tokens')
            if type(emitted) is not int or type(sampled) is not int or not 0 <= emitted <= sampled <= (64 if warmup else 4096):
                checks.append('output_budget')
            elif sampled != emitted + int(row.get('natural_eos') is True):
                checks.append('eos_count')
            if row.get('finish_reason') not in ('eos', 'length'):
                checks.append('finish_reason')
            elif ((row['finish_reason'] == 'eos') != (row.get('natural_eos') is True)
                  or (row['finish_reason'] == 'length' and sampled != (64 if warmup else 4096))):
                checks.append('finish_semantics')
            derived_rate = metrics(row)['useful_decode_tokens_per_second']
            reported_rate = row.get('useful_decode_tokens_per_second')
            if (not finite(derived_rate) or not finite(reported_rate)
                    or not math.isclose(derived_rate, reported_rate, rel_tol=1e-10, abs_tol=1e-10)):
                checks.append('reported_decode_rate')
            counters = row.get('mtp_counters', {})
            if not all(type(counters.get(key)) is int and counters[key] >= 0 for key in WORK):
                checks.append('counter_types')
            elif (counters['accepted'] > counters['proposed'] or counters['proposed'] > 2 * counters['rounds']
                  or counters['verified_rows'] != counters['rounds'] + counters['proposed']):
                checks.append('counter_shape')
            if checks:
                errors.append({'session': index, 'repetition': rep, 'checks': checks})
    integrity['request_grid_complete'] = len(grids) == 4 and all(value['complete'] for value in grids.values())
    integrity['four_post_session_model_guards_pass'] = len(guards) == 4 and all(value is True for value in guards.values())
    integrity['request_checks_pass'] = not errors
    stable = {}
    for warmup, count in ((True, 4), (False, 16)):
        rows = [row for _, _, row in flat if row.get('warmup') is warmup]
        valid_hashes = all(isinstance(row.get(key), str) and len(row[key]) == 64
                           for row in rows for key in ('output_token_ids_sha256', 'text_sha256'))
        stable['warmup' if warmup else 'measured'] = {'requests': len(rows), 'expected_requests': count,
            'unique_output_and_work_signatures': len({signature(row) for row in rows}),
            'matched': len(rows) == count and valid_hashes and len({signature(row) for row in rows}) == 1}
    integrity['all_outputs_lengths_and_work_match'] = all(value['matched'] for value in stable.values())
    pairs = []
    for half, (before_index, candidate_index) in enumerate(PAIRS, 1):
        for rep in range(1, 5):
            before = rows_by_session.get(before_index, {}).get(rep)
            candidate = rows_by_session.get(candidate_index, {}).get(rep)
            if before is None or candidate is None:
                continue
            b, c = metrics(before), metrics(candidate)
            gains, saved = {}, {}
            for metric in METRICS:
                if finite(b[metric]) and finite(c[metric]) and b[metric] > 0:
                    gains[metric] = ((c[metric] - b[metric]) if metric in RATE_METRICS else (b[metric] - c[metric])) / b[metric] * 100
                    if metric not in RATE_METRICS:
                        saved[metric] = b[metric] - c[metric]
                else:
                    gains[metric] = None
            pairs.append({'half': half, 'before_session': before_index, 'candidate_session': candidate_index,
                'repetition': rep, 'output_and_work_match': signature(before) == signature(candidate),
                'improvement_percent': gains, 'absolute_seconds_saved': saved,
                'before_metrics': b, 'candidate_metrics': c})
    integrity['eight_measured_pairs_present'] = len(pairs) == 8
    timing, counts, host, baseline_range = {}, {}, {}, {}
    for arm in ('before', 'candidate'):
        measured = [row for _, name, row in flat if name == arm and row.get('warmup') is False]
        warmups = [row for _, name, row in flat if name == arm and row.get('warmup') is True]
        timing[arm] = {metric: distribution([metrics(row)[metric] for row in measured]) for metric in METRICS}
        counts[arm] = {'measured': request_counts(measured), 'warmups': request_counts(warmups)}
        host[arm] = {'measured': host_summary(measured), 'warmups': host_summary(warmups)}
    for metric in METRICS:
        values = timing['before'][metric]
        baseline_range[metric] = ((values['max'] - values['min']) / values['median'] * 100
                                  if values['n'] and values['median'] > 0 else None)
    gains = {metric: change_summary([pair['improvement_percent'].get(metric) for pair in pairs]) for metric in METRICS}
    saved = {metric: distribution([pair['absolute_seconds_saved'].get(metric) for pair in pairs])
             for metric in METRICS if metric not in RATE_METRICS}
    halves = {str(half): {'measured_pairs': sum(pair['half'] == half for pair in pairs),
        'paired_improvement_percent': {metric: change_summary([pair['improvement_percent'].get(metric)
            for pair in pairs if pair['half'] == half]) for metric in METRICS}}
        for half in (1, 2)}
    finished_grid = manifest.get('completed') is True and integrity['request_grid_complete'] and not missing
    status = 'complete' if all(integrity.values()) else 'review_required' if finished_grid else 'incomplete'
    return {'schema': 'stark-lily-long-context-analysis-v1', 'status': status,
        'attempt': manifest.get('attempt'), 'analyzer_sha256': sha(Path(__file__)),
        'missing_files': missing, 'integrity': integrity, 'request_errors': errors,
        'source_identity_checks': sources, 'session_grids': grids, 'post_session_model_guards': guards,
        'same_seed_output_and_work': stable, 'timing_distributions': timing,
        'paired_improvement_percent': gains, 'paired_absolute_seconds_saved': saved,
        'session_halves': halves, 'paired_rows': pairs, 'request_summaries': counts,
        'host_summaries': host, 'baseline_full_relative_range_percent': baseline_range,
        'scope': 'One whole-book performance case; no automatic promotion, new practical gain gate, benchmark quality score, or general model qualification.',
        'preflight_limit': 'Harness retained planner success only, not numeric planner capacity; completed full-budget generation is the admission evidence.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=HERE / 'run-01')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir)
    output = args.output or args.run_dir / 'analysis.json'
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    temporary.replace(output)
    compact = {'status': result['status'], 'integrity': result.get('integrity'),
        'measured_requests': {arm: values['measured']['requests'] for arm, values in result.get('request_summaries', {}).items()},
        'paired_median_improvement_percent': {metric: values['median'] for metric, values in result.get('paired_improvement_percent', {}).items()},
        'paired_wins': {metric: values['candidate_wins'] for metric, values in result.get('paired_improvement_percent', {}).items()},
        'paired_median_absolute_seconds_saved': {metric: values['median'] for metric, values in result.get('paired_absolute_seconds_saved', {}).items()},
        'request_errors': result.get('request_errors')}
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
