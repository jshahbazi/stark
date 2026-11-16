"""Fetch one pinned public benchmark case in RAM; emit identity/counts only."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path('/Users/johnshahbazian/Library/Application Support/Stark/models/qwen38-flash-next-lily-q4-xl-0f94842e')
HELPER = ROOT / 'runtime/build-candidate/release/stark-template'
GH_REV = '51d9b37b0f1790ead936df2243abbf7f0420e439'
HF_REV = '90f0394333616266d9fe85824ceaf505093cbaa5'
PROMPT_URL = f'https://raw.githubusercontent.com/OpenBMB/InfiniteBench/{GH_REV}/src/prompt.py'
DATA_URL = f'https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/resolve/{HF_REV}/longbook_sum_eng.jsonl'
DATA_SHA = '4e4279cc06c07778f646cda1335755cf5ccdcc38fe9c8a62f1631642fc031027'
DATA_BYTES = 78601868
MIN_INPUT = 65536
MAX_INPUT = 131072

def digest(data):
    return hashlib.sha256(data).hexdigest()

def fetch(url, limit):
    request = urllib.request.Request(url, headers={'User-Agent': 'Stark-public-benchmark-study'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError('source_too_large')
    return data

def prepare():
    sys.path[:0] = [str(ROOT / 'tools/converter'), str(ROOT / 'scripts')]
    from stark_convert import chat, reasoning, template_runtime
    from tokenizers import Tokenizer
    os.environ['PATH'] = str(HELPER.parent) + os.pathsep + os.environ.get('PATH', '')
    template_runtime._template_executable = lambda: HELPER.resolve()
    codec, chat_template = chat.load_template(MODEL, expected_hashes={
        'config_sha256': digest((MODEL / 'config.json').read_bytes()),
        'template_sha256': digest((MODEL / 'chat_template.jinja').read_bytes())})
    source = fetch(PROMPT_URL, 65536)
    module = ast.parse(source)
    assignment = next(node for node in module.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'gpt4_templates' for target in node.targets))
    upstream_template = ast.literal_eval(assignment.value)['longbook_sum_eng']
    raw = fetch(DATA_URL, DATA_BYTES)
    if len(raw) != DATA_BYTES or digest(raw) != DATA_SHA:
        raise RuntimeError('dataset_integrity_failed')
    tokenizer = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    inspected = []
    for index, line in enumerate(raw.splitlines()):
        example = json.loads(line)
        context = example['context']
        if not isinstance(context, str):
            raise RuntimeError('dataset_context_invalid')
        prompt = upstream_template.format(context=context)
        messages = ([{'role': 'system', 'content': codec['default_system_prompt']}] if codec.get('default_system_prompt') else [])
        messages.append({'role': 'user', 'content': prompt})
        rendered = chat.render_template(chat_template, messages=messages, tools=None,
            add_generation_prompt=True, **reasoning.reasoning_options(codec, thinking=False),
            bos_token=codec.get('bos_token') or '', eos_token=codec.get('eos_token') or '')
        ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        inspected.append({'row_index_zero_based': index, 'rendered_input_tokens': len(ids), 'record_sha256': digest(line)})
        if MIN_INPUT <= len(ids) <= MAX_INPUT:
            identity = {
                'benchmark': 'InfiniteBench', 'task': 'longbook_sum_eng', 'prompt_variant': 'gpt4_templates',
                'source_github_revision': GH_REV, 'dataset_revision': HF_REV,
                'prompt_source_url': PROMPT_URL, 'prompt_source_sha256': digest(source),
                'prompt_template_sha256': digest(upstream_template.encode()),
                'dataset_url': DATA_URL, 'dataset_sha256': DATA_SHA, 'dataset_bytes': len(raw),
                'row_index_zero_based': index, 'record_sha256': digest(line),
                'dataset_id': example.get('id') if type(example.get('id')) is int else None,
                'source_context_sha256': digest(context.encode()), 'source_context_utf8_bytes': len(context.encode()),
                'prompt_sha256': digest(prompt.encode()), 'rendered_sha256': digest(rendered.encode()),
                'rendered_utf8_bytes': len(rendered.encode()), 'input_tokens': len(ids),
                'input_token_ids_sha256': digest(b''.join(i.to_bytes(4,'little') for i in ids)),
                'selection': 'first original-order whole record with 65536..131072 native rendered tokens; no padding/truncation or performance-based selection',
                'inspected_length_candidates': inspected,
                'tokenizer_json_sha256': digest((MODEL / 'tokenizer.json').read_bytes()),
                'template_helper_sha256': digest(HELPER.read_bytes()),
                'codec_sha256': digest(json.dumps(codec,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()),
                'context_modified': False, 'prompt_modified': False,
                'reference_summary_used': False,
            }
            return rendered, ids, identity
    raise RuntimeError('no_record_in_length_range')

if __name__ == '__main__':
    try:
        _, _, identity = prepare()
        target = Path(__file__).with_name('preparation.json')
        if target.exists():
            raise RuntimeError('identity_exists')
        target.write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n')
        print(json.dumps(identity,indent=2,sort_keys=True))
    except BaseException as error:
        print(json.dumps({'status':'failed','failure_type':type(error).__name__}))
        raise SystemExit(2)
