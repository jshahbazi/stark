"""Publish aggregate verifier evidence and immutable experimental source only."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT / 'docs/archive/experiments/2026-11-15-lily-verifier'
DATA = ROOT / 'docs/benchmarks/data/lily-small-row-verifier-2026-11-15.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def main():
    comparison = read(HERE / 'compare-v4/analysis.json')
    screen = read(HERE / 'screen-v2/analysis.json')
    if comparison.get('status') != 'complete' or screen.get('status') != 'complete':
        raise RuntimeError('analysis_not_complete')
    if DATA.exists() or ARCHIVE.exists():
        raise RuntimeError('publication_exists')
    private_hashes = {}
    for dirname in ('screen', 'screen-v2', 'detailed-v3', 'compare-v4'):
        for path in sorted((HERE / dirname).glob('*')):
            if path.is_file() and path.suffix in {'.json', '.jsonl'}:
                private_hashes[str(path.relative_to(HERE))] = sha(path)
    rows = {}
    for dirname, count in (('screen', 1), ('screen-v2', 4), ('detailed-v3', 2), ('compare-v4', 4)):
        rows[dirname] = []
        for index in range(1, count + 1):
            path = HERE / dirname / f'session-{index}.json'
            session = read(path)
            for row in session['requests']:
                allowed = ('case', 'warmup', 'repetition', 'request_status', 'input_tokens',
                    'prefix_reused_tokens', 'prefix_reuse_valid', 'output_tokens', 'sampled_tokens',
                    'finish_reason', 'natural_eos', 'output_token_ids_sha256', 'text_sha256',
                    'counts_valid', 'automatic_depth_valid', 'actual_drafts', 'mtp_counters',
                    'mtp_verification', 'decode_seconds', 'generation_seconds', 'prefill_seconds',
                    'useful_decode_tokens_per_second', 'timings_valid', 'wall_seconds', 'host',
                    'started_utc', 'finished_utc', 'transport_failure_type', 'response_structure')
                rows[dirname].append({'session': index, 'arm': session['arm'],
                    **{k: row[k] for k in allowed if k in row}})
    source_names = {
        'run-invalid-selector.py': 'run-invalid-selector-as-run.py',
        'run-screen-v2-as-run.py': 'run-screen-as-run.py',
        'run-detailed-v3-as-run.py': 'run-detailed-as-run.py',
        'run.py': 'run-comparison-as-run.py',
        'analyze_screen.py': 'analyze_screen.py',
        'analyze_compare.py': 'analyze_compare.py',
        'diagnostic-stage-attribution.patch': 'diagnostic-stage-attribution.patch',
        'q8-f32-head-reuse-diagnostic.patch': 'head-only-prepared-unbuilt.patch',
        'generic-projection-reuse-diagnostic-v1.patch': 'projection-reuse-v1-unbuilt.patch',
        'generic-projection-reuse-diagnostic.patch': 'projection-reuse-v2-as-run.patch',
        'prepare_q8_head_candidate.py': 'prepare_head_candidate.py',
        'prepare_projection_reuse_candidate-v1.py': 'prepare_projection_reuse_v1.py',
        'prepare_projection_reuse_candidate.py': 'prepare_projection_reuse_v2.py',
        'publish.py': 'publish.py',
    }
    ARCHIVE.mkdir(parents=True)
    source_hashes = {}
    for origin, name in source_names.items():
        path = HERE / origin
        shutil.copyfile(path, ARCHIVE / name)
        source_hashes[name] = sha(path)
    manifests = {name: read(HERE / name / 'manifest.json')
                 for name in ('screen', 'screen-v2', 'detailed-v3', 'compare-v4')}
    # Session arrays are redundant with the whitelisted per-request records;
    # retain the content-free configuration and source seals here.
    for manifest in manifests.values():
        manifest.pop('sessions', None)
    gates = {}
    for case in ('coding', 'planning'):
        metrics = comparison['same_code_baseline_range_and_hurdle'][case]
        changes = comparison['paired_changes'][case]['measured_pair_changes']
        gates[case] = any(metrics[metric]['hurdle_met'] and changes[metric]['candidate_wins'] >= 7
                          for metric in ('wall_seconds', 'useful_decode_tokens_per_second'))
    performance_passed = all(gates.values())
    data = {
        'schema': 'stark-lily-verifier-public-aggregate-v1',
        'attempts': ['20261115-LILY-VERIFY-01', '20261115-LILY-VERIFY-02',
                     '20261115-LILY-VERIFY-03', '20261115-LILY-VERIFY-04'],
        'privacy': 'Aggregate counts, timings, hashes, fixed categories and source only. Prompt, response, vectors and probabilities were never persisted. Raw aggregate native profiles remain private.',
        'build_identity': read(HERE / 'build-identity.json'),
        'performance_screening': {'passed': performance_passed, 'per_prompt_passed': gates,
            'disposition': 'awaiting full qualification' if performance_passed else 'rejected for production promotion; practical gain criterion failed'},
        'configuration': manifests,
        'coarse_profile_analysis': screen,
        'comparison_analysis': comparison,
        'requests': rows,
        'private_receipt_sha256': private_hashes,
        'archived_source_sha256': source_hashes,
        'historical_results': 'Original MTP depth-policy and natural-response data are untouched. This is a separate two-draft engine experiment.',
    }
    DATA.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'data_sha256': sha(DATA), 'source_files': len(source_hashes),
                      'private_receipts': len(private_hashes)}))

if __name__ == '__main__':
    main()
