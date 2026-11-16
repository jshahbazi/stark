"""Publish aggregate-only long-context evidence and byte-identical sources."""
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ARCHIVE=ROOT/'docs/archive/experiments/2026-11-15-lily-long-context'
DATA=ROOT/'docs/benchmarks/data/lily-long-context-infinitebench-2026-11-15.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def main():
    run=HERE/'run-01'
    analysis=read(run/'analysis.json')
    manifest=read(run/'manifest.json')
    if analysis.get('status')!='complete' or manifest.get('completed') is not True:
        raise RuntimeError('incomplete_run')
    if ARCHIVE.exists() or DATA.exists():
        raise RuntimeError('publication_exists')
    ARCHIVE.mkdir(parents=True)
    archived={}
    for name in ('prepare.py','run.py','analyze.py','publish.py'):
        target=ARCHIVE/('run-as-run.py' if name=='run.py' else name)
        shutil.copyfile(HERE/name,target)
        archived[target.name]=sha(target)
    requests=[]
    sessions=[]
    for index in range(1,5):
        session=read(run/f'session-{index}.json')
        sessions.append({k:v for k,v in session.items() if k!='requests'})
        for row in session['requests']:
            fields=('repetition','warmup','wall_seconds','host','started_utc','finished_utc',
                'progress_events','transport_failure_type','request_status','input_tokens',
                'prefix_reused_tokens','prefix_reuse_valid','output_tokens','sampled_tokens',
                'finish_reason','natural_eos','output_token_ids_sha256','text_sha256',
                'counts_valid','automatic_depth_valid','actual_drafts','mtp_counters','mtp_verification',
                'decode_seconds','generation_seconds','prefill_seconds','useful_decode_tokens_per_second',
                'timings_valid','response_structure','same_work_and_output')
            requests.append({'session':index,'arm':session['arm'],
                **{key:row[key] for key in fields if key in row}})
    configuration=dict(manifest)
    configuration.pop('sessions',None)
    receipts={}
    for path in (HERE/'preparation.json',HERE/'source-provenance.json',HERE/'machine.json',*sorted(run.glob('*.json'))):
        receipts[str(path.relative_to(HERE))]=sha(path)
    data={
        'schema':'stark-lily-long-context-public-aggregate-v1','attempt':manifest['attempt'],
        'configuration':configuration,'source_provenance':read(HERE/'source-provenance.json'),
        'machine':read(HERE/'machine.json'),
        'analysis':analysis,'sessions':sessions,'requests':requests,
        'private_receipt_sha256':receipts,'archived_source_sha256':archived,
        'prior_experiment':{'source_base':'2ac1b713d893f85b97111c1279465cfaa12d7bbe',
            'source_report':'2026-11-15-lily-small-row-verifier.md',
            'candidate_patch_sha256':'9f406ddd07961d1d529622d13699cb119aacdbf9dbeba33fef0d6e4e5e4b0768',
            'component_control':'Historical unchanged-build control: 60 cases, 9375336 bytes, zero unequal bytes. Not rerun or broadened.'},
        'scope':'One unmodified public whole-book performance case, not an official InfiniteBench suite or quality score.',
        'promotion':'Additional evidence only; source remains experimental, production default unchanged; not installed/released.',
        'privacy':'Only hashes,counts,timings,fixed metadata and original harness source are published. Corpus/rendered inputs/outputs/vectors/probabilities never persisted.',
    }
    DATA.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'data_sha256':sha(DATA),'archived_sources':len(archived),
        'private_receipts':len(receipts),'requests':len(requests)}))

if __name__=='__main__':
    main()
