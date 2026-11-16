"""Matched long-context performance case; input/output content stays in RAM."""
from __future__ import annotations
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
OUT = HERE / 'run-01'
BASE_SHA = '61b9c8cc724407573169eb2abad7f908e4943eead7ac3122eeec3ef841a48640'
CANDIDATE_SHA = '096cceb6b370fea71b49523c928c68c1ac172ca77b63a968acf7ceb0a26dbe2e'
PATCH_SHA = '9f406ddd07961d1d529622d13699cb119aacdbf9dbeba33fef0d6e4e5e4b0768'
MAX_NEW = 4096
REQUEST_DEADLINE = 1800
SAMPLING = {'mode':'stochastic','temperature':1.0,'top_k':20,'top_p':0.95}
ORDER = ['before','candidate','candidate','before']

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def run():
    os.umask(0o077)
    if OUT.exists():
        raise RuntimeError('output_exists')
    sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tools/converter')]
    prior=load('_long_helpers',ROOT/'runtime/lily-slack-long/run.py')
    identities=load('_identity_helpers',prior.OLD_RUNNER)
    prepared=load('_long_context_prepare',HERE/'prepare.py')
    candidate=ROOT/'runtime/lily-verifier-study/candidate-source/target/release/stark-generate'
    patch=ROOT/'docs/archive/experiments/2026-11-15-lily-verifier/projection-reuse-v2-as-run.patch'
    model_identity=json.loads(prior.IDENTITY.read_text())
    if (sha(prior.BASE)!=BASE_SHA or sha(candidate)!=CANDIDATE_SHA or sha(patch)!=PATCH_SHA
            or not identities.current_model_identity(model_identity)):
        raise RuntimeError('sealed_identity_mismatch')
    for key in tuple(os.environ):
        if key.startswith(('STARK_','UZU_','LILY_','GGML_')):
            os.environ.pop(key)
    os.environ['STARK_REQUEST_BUDGET_MIB']='16'
    rendered,ids,source=prepared.prepare()
    if source != json.loads((HERE/'preparation.json').read_text()):
        raise RuntimeError('public_case_identity_changed')
    context=((len(ids)+MAX_NEW+8191)//8192)*8192
    if context>262144:
        raise RuntimeError('context_limit')
    from validate_generation_prefix_cache import Worker
    from benchmark_clean_qwen import Monitor,paging
    from benchmark_glm_mlx_serve import screen_attempt

    class ProgressWorker(Worker):
        def request(self,payload,**kwargs):
            if payload.get('operation')!='generate':
                return super().request(payload,**kwargs)
            self.request_id+=1
            started=time.monotonic()
            self.send(dict(payload,request_id=self.request_id))
            deadline=started+REQUEST_DEADLINE
            count=0
            last_print=started
            last_sampled=0
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    raise TimeoutError('request_deadline')
                self.timeout=min(900,remaining)
                reply=self.read()
                if reply.get('event')=='delta':
                    continue
                if reply.get('event')=='progress':
                    count+=1
                    sampled=reply.get('sampled_tokens',0)
                    if type(sampled) is int:
                        last_sampled=sampled
                    now=time.monotonic()
                    if now-last_print>=20:
                        print(json.dumps({'event':'progress','session':self.study_index,
                            'arm':self.study_arm,'repetition':self.study_repetition,
                            'elapsed_seconds':round(now-started,1),'sampled_tokens':last_sampled,
                            'prefill_seconds':reply.get('prefill_seconds')}),flush=True)
                        last_print=now
                    continue
                reply['_progress_count']=count
                return reply

    OUT.mkdir(parents=True)
    manifest={
        'schema':'stark-lily-infinitebench-long-context-v1','attempt':'20261115-LILY-LONGCTX-01',
        'started_utc':utc(),'source_checkout_commit':'a4a1cc3021281a329cd42f0f7bf350a14a25bee1',
        'worker_source_base':'2ac1b713d893f85b97111c1279465cfaa12d7bbe',
        'baseline_source':'cc8e47a824f802e24d442183ebbdd29fd25ecaaa',
        'worker_sha256':{'before':BASE_SHA,'candidate':CANDIDATE_SHA},'candidate_patch_sha256':PATCH_SHA,
        'harness_sha256':sha(Path(__file__)),'prepare_sha256':sha(HERE/'prepare.py'),
        'generation_helper_sha256':sha(ROOT/'runtime/lily-slack-long/run.py'),
        'model_identity_helper_sha256':sha(prior.OLD_RUNNER),
        'model_identity_receipt_sha256':sha(prior.IDENTITY),
        'model_content_manifest_sha256':model_identity.get('model_content_manifest_sha256'),
        'native_source_sha256':sha(prior.MODEL/'native-source.json'),
        'config_sha256':sha(prior.MODEL/'config.json'),'source':source,
        'selection_policy':'one pinned public whole-book record; unmodified official prompt; task/performance case only, not full InfiniteBench score',
        'context_tokens':context,'max_new_tokens':MAX_NEW,'warmup_max_new_tokens':64,
        'repetitions_per_session':4,'expected_measured_requests':16,'expected_warmups':4,
        'session_order':ORDER,'sampling':SAMPLING,'seed':42,'mtp_drafts':2,'stream':True,
        'vision_loaded':True,'prefix_cache':False,'expected_reused_input_tokens':0,
        'request_deadline_seconds':REQUEST_DEADLINE,'inactivity_timeout_seconds':900,
        'host_policy':'all observations retained under current load; no quiet gate, cooling, retries or exclusions',
        'warmup_policy':'one same-book full-context request per worker,64 output tokens; excluded only as preregistered warmup',
        'identity_policy':'guard once before run and after every session; separate worker executable hashes checked before each load',
        'comparison_policy':'pairs before1/candidate2 and before4/candidate3 by repetition; report prefill, decode and wall separately; require same seeded outputs and MTP work for scheduling attribution; retain modest gains as evidence',
        'promotion_policy':'no automatic promotion from a single public case; prior practical criteria and full correctness/lifecycle qualification remain separate',
        'privacy':'dataset and rendered input/output/token arrays held only in memory; persist hashes,counts,timings and fixed categories only',
        'runtime':{'python':platform.python_version(),'platform':platform.platform()},
        'sessions':[],'completed':False,
    }
    prior.write(OUT/'manifest.json',manifest)
    print(json.dumps({'event':'prepared','input_tokens':len(ids),'context_tokens':context,
        'output_ceiling':MAX_NEW,'dataset_row':source['row_index_zero_based']}),flush=True)
    reference={}
    for index,arm in enumerate(ORDER,1):
        binary=prior.BASE if arm=='before' else candidate
        if sha(binary)!=manifest['worker_sha256'][arm]:
            raise RuntimeError('worker_changed')
        if arm=='candidate':
            os.environ['STARK_LILY_VERIFY_PROJECTION_REUSE']='1'
        else:
            os.environ.pop('STARK_LILY_VERIFY_PROJECTION_REUSE',None)
        session={'index':index,'arm':arm,'started_utc':utc(),'requests':[],'failure_type':None}
        manifest['sessions'].append(session)
        worker=None
        monitor=None
        try:
            worker=ProgressWorker(str(binary),str(prior.MODEL),prefix_cache=False,timeout=900,vision=True)
            worker.study_index=index
            worker.study_arm=arm
            ready=worker.ready
            mtp=ready.get('mtp') or {}
            image=ready.get('image_input') or {}
            if (mtp.get('recommended_drafts')!=2 or mtp.get('supported') is not True
                    or image.get('family_profile')!=prior.IMAGE_PROFILE or ready.get('max_context_tokens',0)<context):
                raise RuntimeError('readiness_mismatch')
            session['ready']={'mtp_recommended_drafts':mtp.get('recommended_drafts'),
                'image_family_profile':image.get('family_profile'),'max_context_tokens':ready.get('max_context_tokens')}
            native_ids=worker.tokenize(rendered)
            if native_ids!=ids:
                raise RuntimeError('native_tokenizer_mismatch')
            session['input_tokenization_match']=True
            plan=worker.request({'operation':'plan_generation','input_tokens':len(ids),
                'context_tokens':context,'mtp_drafts':2,'sampling':SAMPLING})
            session['preflight_plan_ok']='error' not in plan
            if 'error' in plan:
                session['preflight_error']=plan['error'] if isinstance(plan['error'],str) and plan['error'].replace('_','').isalnum() else 'unclassified'
                raise RuntimeError('plan_rejected')
            monitor=Monitor().__enter__()
            for rep in range(5):
                worker.study_repetition=rep
                limit=64 if rep==0 else MAX_NEW
                request={'operation':'generate','rendered_prompt':rendered,'context_tokens':context,
                    'max_new_tokens':limit,'seed':42,'sampling':SAMPLING,'mtp_drafts':2,'stream':True}
                reply,wall,started,finished,host,transport_failure=prior.monitor_request(
                    worker,request,monitor,f'{index}-book-{rep}',paging(),paging,screen_attempt)
                progress_count=reply.pop('_progress_count',0) if isinstance(reply,dict) else 0
                row,text=prior.validate_generation(reply,ready,'before',len(ids),limit)
                row.update({'repetition':rep,'warmup':rep==0,'wall_seconds':wall,'host':host,
                    'started_utc':started,'finished_utc':finished,'progress_events':progress_count,
                    'transport_failure_type':transport_failure})
                if text is not None:
                    row['response_structure']=prior.response_structure(text)
                text=None
                reply=None
                key=rep==0
                signature={k:row.get(k) for k in ('output_token_ids_sha256','text_sha256',
                    'output_tokens','sampled_tokens','finish_reason','mtp_counters')}
                row['same_work_and_output']=signature==reference.setdefault(key,signature)
                session['requests'].append(row)
                prior.write(OUT/f'session-{index}.json',session)
                print(json.dumps({'event':'request_complete','session':index,'arm':arm,'repetition':rep,
                    'status':row.get('request_status'),'input_tokens':row.get('input_tokens'),
                    'output_tokens':row.get('output_tokens'),'prefill_seconds':row.get('prefill_seconds'),
                    'decode_seconds':row.get('decode_seconds'),'wall_seconds':round(wall,3),
                    'same_work_and_output':row['same_work_and_output']}),flush=True)
                if (transport_failure or row.get('request_status')!='ok'
                        or row.get('mtp_verification')!='batched-decode-exact-v1'
                        or not all(row.get(k) for k in ('counts_valid','automatic_depth_valid','timings_valid',
                            'prefix_reuse_valid','same_work_and_output'))):
                    raise RuntimeError('request_validation_failed')
        except BaseException as error:
            session['failure_type']=type(error).__name__
        finally:
            if monitor is not None:
                monitor.__exit__(None,None,None)
            if worker is not None:
                worker.close()
        session['finished_utc']=utc()
        session['model_guard_passed']=identities.current_model_identity(model_identity)
        prior.write(OUT/f'session-{index}.json',session)
        prior.write(OUT/'manifest.json',manifest)
        if session['failure_type'] or not session['model_guard_passed']:
            print(json.dumps({'event':'session_failed','session':index,'failure_type':session['failure_type']}),flush=True)
            return 2
    manifest['completed']=True
    manifest['finished_utc']=utc()
    prior.write(OUT/'manifest.json',manifest)
    return 0

if __name__=='__main__':
    try:
        result=run()
    except BaseException as error:
        print(json.dumps({'status':'failed','failure_type':type(error).__name__}),flush=True)
        result=2
    raise SystemExit(result)
