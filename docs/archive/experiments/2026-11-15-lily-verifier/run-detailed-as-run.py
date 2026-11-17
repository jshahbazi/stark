"""Private aggregate-only two-draft natural-response verifier study."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['screen', 'detailed', 'compare'], required=True)
    parser.add_argument('--diagnostic', type=Path)
    parser.add_argument('--diagnostic-sha256')
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--candidate-sha256')
    args = parser.parse_args()
    os.umask(0o077)
    out = HERE / (args.phase + '-v3')
    if out.exists():
        raise RuntimeError('output_exists')
    sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tools/converter')]
    prior = load('_long_helpers', ROOT / 'runtime/lily-slack-long/run.py')
    identity_helpers = load('_identity_helpers', prior.OLD_RUNNER)
    identity = json.loads(prior.IDENTITY.read_text())
    if not identity_helpers.current_model_identity(identity):
        raise RuntimeError('model_identity_changed')
    if digest(prior.BASE) != prior.BASE_SHA256:
        raise RuntimeError('baseline_identity_changed')
    if args.phase == 'compare' and (not args.candidate or digest(args.candidate) != args.candidate_sha256):
        raise RuntimeError('candidate_identity_missing')
    if args.phase == 'detailed' and (not args.diagnostic or digest(args.diagnostic) != args.diagnostic_sha256):
        raise RuntimeError('diagnostic_identity_missing')
    from stark_convert import chat, reasoning, template_runtime
    from validate_generation_prefix_cache import Worker
    from benchmark_clean_qwen import Monitor, paging
    from benchmark_glm_mlx_serve import screen_attempt
    for key in tuple(os.environ):
        if key.startswith(('STARK_', 'UZU_', 'LILY_', 'GGML_')):
            os.environ.pop(key)
    os.environ['STARK_REQUEST_BUDGET_MIB'] = '16'
    os.environ['PATH'] = str(prior.TEMPLATE_HELPER.parent) + os.pathsep + os.environ.get('PATH', '')
    template_runtime._template_executable = lambda: prior.TEMPLATE_HELPER.resolve()
    template_hash = digest(prior.MODEL / 'chat_template.jinja')
    codec, template = chat.load_template(prior.MODEL, expected_hashes={
        'config_sha256': prior.CONFIG_SHA256, 'template_sha256': template_hash})
    if sys.stdin.isatty():
        import termios
        flags = termios.tcgetattr(sys.stdin.fileno())
        flags[3] &= ~termios.ECHO
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, flags)
    print('ready_for_prompts', flush=True)
    supplied = json.loads(sys.stdin.readline())
    if set(supplied) != {'coding', 'planning'} or not all(isinstance(v, str) and v.strip() for v in supplied.values()):
        raise RuntimeError('invalid_input')
    cases = [('coding', supplied['coding'], 2048), ('planning', supplied['planning'], 4096)]
    order = (['off', 'stage', 'stage', 'off'] if args.phase == 'screen' else
             ['stage', 'stage'] if args.phase == 'detailed' else ['before', 'candidate', 'candidate', 'before'])
    repetitions = 4 if args.phase == 'compare' else 1
    out.mkdir(parents=True)
    manifest = {
        'schema': 'stark-lily-small-row-verifier-study-v1', 'phase': args.phase,
        'attempt': {'screen': '20261115-LILY-VERIFY-01', 'detailed': '20261115-LILY-VERIFY-02', 'compare': '20261115-LILY-VERIFY-04'}[args.phase],
        'started_utc': utc(), 'source_commit': '2ac1b713d893f85b97111c1279465cfaa12d7bbe',
        'worker_sha256': {'baseline': digest(prior.BASE), 'candidate': digest(args.candidate) if args.candidate else None,
                         'diagnostic': digest(args.diagnostic) if args.diagnostic else None},
        'harness_sha256': digest(Path(__file__)), 'helper_sha256': digest(ROOT / 'runtime/lily-slack-long/run.py'),
        'identity_helper_sha256': digest(prior.OLD_RUNNER), 'model_identity_receipt_sha256': digest(prior.IDENTITY),
        'model_content_manifest_sha256': identity.get('model_content_manifest_sha256'),
        'template_helper_sha256': digest(prior.TEMPLATE_HELPER), 'template_sha256': template_hash,
        'native_source_sha256': digest(prior.MODEL / 'native-source.json'),
        'config_sha256': digest(prior.MODEL / 'config.json'),
        'model_guard_files': 72, 'image_profile': prior.IMAGE_PROFILE,
        'sampling': dict(prior.SAMPLING), 'seed': 42, 'mtp': 2, 'prefix_cache': False,
        'context_tokens': 8192, 'session_order': order, 'repetitions_per_case_per_session': repetitions,
        'warmup_max_new_tokens': 64, 'host_policy': 'busy host; all observations retained; no quiet gate or retry',
        'screen_policy': 'stage timestamps attribute inclusive GPU spans; off sessions establish uninstrumented timing; do not interpret profile/off as optimization',
        'promotion_policy': 'only prototype a diagnosed stage with credible >=10% overall gain; require >=10% paired throughput or latency gain beyond same-code variation, >=7/8 paired wins per prompt, no >5% regression; same seeded outputs and MTP counts for scheduling-only change',
        'cases': {label: {'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'prompt_utf8_bytes': len(prompt.encode()), 'max_new_tokens': ceiling} for label, prompt, ceiling in cases},
        'runtime': {'python': platform.python_version(), 'platform': platform.platform()},
        'sessions': [], 'completed': False,
    }
    prior.write(out / 'manifest.json', manifest)
    for index, arm in enumerate(order, 1):
        stage = arm == 'stage'
        if stage:
            os.environ['STARK_QWEN38_PROFILE_OUTPUT'] = str(out / f'profile-{index}.jsonl')
        else:
            os.environ.pop('STARK_QWEN38_PROFILE_OUTPUT', None)
        binary = (args.diagnostic if args.phase == 'detailed' else
                  args.candidate if arm == 'candidate' else prior.BASE)
        session = {'index': index, 'arm': arm, 'started_utc': utc(), 'requests': [], 'failure_type': None}
        manifest['sessions'].append(session)
        worker = None
        monitor = None
        try:
            worker = Worker(str(binary), str(prior.MODEL), prefix_cache=False, timeout=900, vision=True)
            ready = worker.ready
            mtp = ready.get('mtp') or {}
            image = ready.get('image_input') or {}
            if (mtp.get('supported') is not True or mtp.get('recommended_drafts') != 2
                    or image.get('family_profile') != prior.IMAGE_PROFILE):
                raise RuntimeError('worker_readiness_mismatch')
            session['ready'] = {'mtp_recommended_drafts': mtp.get('recommended_drafts'),
                'image_family_profile': image.get('family_profile'), 'image_verification': image.get('verification')}
            monitor = Monitor().__enter__()
            case_order = cases if index % 2 else list(reversed(cases))
            for label, prompt, ceiling in case_order:
                rendered, ids, codec_hash = prior.render_and_tokenize(prompt, worker, codec, template, chat, reasoning)
                session.setdefault('inputs', {})[label] = {'input_tokens': len(ids),
                    'codec_sha256': codec_hash, 'rendered_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
                    'token_ids_sha256': hashlib.sha256(b''.join(t.to_bytes(4, 'little') for t in ids)).hexdigest()}
                for repetition in range(repetitions + 1):
                    limit = 64 if repetition == 0 else ceiling
                    payload = {'operation': 'generate', 'rendered_prompt': rendered, 'context_tokens': 8192,
                        'max_new_tokens': limit, 'sampling': prior.SAMPLING, 'seed': 42, 'mtp_drafts': 2}
                    reply, wall, started, finished, host, failure = prior.monitor_request(
                        worker, payload, monitor, f'{index}-{label}-{repetition}', paging(), paging, screen_attempt)
                    row, text = prior.validate_generation(reply, ready, 'before', len(ids), limit)
                    row.update({'case': label, 'warmup': repetition == 0, 'repetition': repetition,
                        'wall_seconds': wall, 'started_utc': started, 'finished_utc': finished, 'host': host,
                        'transport_failure_type': failure})
                    if text is not None:
                        row['response_structure'] = prior.response_structure(text)
                    text = None
                    reply = None
                    if stage:
                        profiles = [json.loads(line) for line in (out / f'profile-{index}.jsonl').read_text().splitlines()]
                        row['aggregate_profile'] = profiles[-1] if profiles else None
                    session['requests'].append(row)
                    prior.write(out / f'session-{index}.json', session)
                    print(json.dumps({'session': index, 'arm': arm, 'case': label, 'warmup': repetition == 0,
                        'status': row.get('request_status'), 'emitted': row.get('output_tokens'),
                        'wall_seconds': round(wall, 3), 'depth': row.get('actual_drafts')}), flush=True)
                    if (failure or row.get('request_status') != 'ok' or row.get('mtp_verification') != 'batched-decode-exact-v1' or not all(row.get(k) for k in
                            ('counts_valid', 'automatic_depth_valid', 'timings_valid', 'prefix_reuse_valid'))):
                        raise RuntimeError('request_validation_failed')
        except BaseException as error:
            session['failure_type'] = type(error).__name__
        finally:
            if monitor:
                monitor.__exit__(None, None, None)
            if worker:
                worker.close()
        session['finished_utc'] = utc()
        session['model_guard_passed'] = identity_helpers.current_model_identity(identity)
        prior.write(out / f'session-{index}.json', session)
        prior.write(out / 'manifest.json', manifest)
        if session['failure_type'] or not session['model_guard_passed']:
            print(json.dumps({'status': 'session_failed', 'session': index, 'failure_type': session['failure_type']}), flush=True)
            return 2
    manifest['completed'] = True
    manifest['finished_utc'] = utc()
    prior.write(out / 'manifest.json', manifest)
    return 0

if __name__ == '__main__':
    try:
        result = run()
    except BaseException as error:
        print(json.dumps({'status': 'failed', 'failure_type': type(error).__name__}), flush=True)
        result = 2
    raise SystemExit(result)
