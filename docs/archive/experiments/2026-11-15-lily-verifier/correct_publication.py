"""Correct coarse-profile pair labels; preserve the initial publication."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'docs/benchmarks/data/lily-small-row-verifier-2026-11-15.json'
ARCHIVE = ROOT / 'docs/archive/experiments/2026-11-15-lily-verifier'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    initial_sha = sha(DATA)
    assert initial_sha == '89a5f609d07a513eb5577b7424722d730d845847caf52a85b614462d283b96ac'
    original = json.loads(DATA.read_text())
    corrected_path = HERE / 'screen-v2/analysis-label-corrected.json'
    corrected = json.loads(corrected_path.read_text())
    old = original['coarse_profile_analysis']
    assert {k:v for k,v in old.items() if k != 'matched_off_stage_requests'} == {
        k:v for k,v in corrected.items() if k != 'matched_off_stage_requests'}
    for before, after in zip(old['matched_off_stage_requests'], corrected['matched_off_stage_requests'], strict=True):
        if before['off_session'] == 3:
            expected = dict(before, off_session=4, stage_session=3,
                            off=before['stage'], stage=before['off'])
            assert after == expected
        else:
            assert before == after
    retained = HERE / 'publication-initial.json'
    assert not retained.exists()
    shutil.copyfile(DATA, retained)
    shutil.copyfile(HERE / 'analyze_screen_initial.py', ARCHIVE / 'analyze_screen_initial.py')
    shutil.copyfile(HERE / 'analyze_screen.py', ARCHIVE / 'analyze_screen.py')
    shutil.copyfile(Path(__file__), ARCHIVE / 'correct_publication.py')
    original['coarse_profile_analysis'] = corrected
    original['analysis_correction'] = {
        'scope': 'Coarse screen second-half off/stage session labels and nested roles only; numerical results and all native receipts unchanged',
        'initial_publication_sha256': initial_sha,
        'initial_analysis_retained': 'screen-v2/analysis.json',
        'corrected_analysis': 'screen-v2/analysis-label-corrected.json',
        'initial_analyzer_sha256': sha(HERE / 'analyze_screen_initial.py'),
        'corrected_analyzer_sha256': sha(HERE / 'analyze_screen.py'),
        'pair_order': [[1, 2], [4, 3]],
    }
    original['private_receipt_sha256']['screen-v2/analysis-label-corrected.json'] = sha(corrected_path)
    original['private_receipt_sha256']['publication-initial.json'] = sha(retained)
    for name in ('analyze_screen.py', 'analyze_screen_initial.py', 'correct_publication.py'):
        original['archived_source_sha256'][name] = sha(ARCHIVE / name)
    DATA.write_text(json.dumps(original, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'final_data_sha256': sha(DATA), 'native_measurements_changed': False}))

if __name__ == '__main__':
    main()
