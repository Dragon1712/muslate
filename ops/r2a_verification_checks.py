"""Negative regression checks for the supplemental artifact verifier/report guard."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_r2a_artifacts import verify
from summarize_r2a import generate


def checks(root,source):
    root=Path(root);results=[]
    with tempfile.TemporaryDirectory(prefix='r2a-verify-',dir=root.parent) as temporary:
        fixture=Path(temporary);shutil.copy2(root/'protocol.json',fixture/'protocol.json')
        shutil.copytree(root/'cells',fixture/'cells')
        first=sorted((fixture/'cells').glob('*.json'))[0];original=json.loads(first.read_text())
        mislabeled=json.loads(json.dumps(original));mislabeled['condition']='L1'
        first.write_text(json.dumps(mislabeled))
        try:verify(fixture,source)
        except ValueError as e:
            if str(e)!='cell label/stem mismatch':raise
            results.append(dict(check='mislabeled cell rejected',error=str(e)))
        else:raise AssertionError('mislabeled cell accepted')
        swapped=json.loads(json.dumps(original));swapped['identity']['training_identity']['config']['arm']['seed']=2
        first.write_text(json.dumps(swapped))
        try:verify(fixture,source)
        except ValueError as e:
            if str(e)!='training arm identity':raise
            results.append(dict(check='swapped checkpoint identity rejected',error=str(e)))
        else:raise AssertionError('swapped identity accepted')
        # A minimal reporting input demonstrates failure before report generation.
        reporting=fixture/'reporting';reporting.mkdir();protocol=reporting/'protocol.json'
        protocol.write_bytes((root/'protocol.json').read_bytes())
        sha=hashlib.sha256(protocol.read_bytes()).hexdigest()
        (reporting/'verified_manifest.json').write_text(json.dumps(dict(status='verified',input_sha256={'protocol.json':sha})))
        protocol.write_bytes(protocol.read_bytes()+b'\n')
        try:generate(reporting,reporting/'should_not_exist.md')
        except ValueError as e:
            if str(e)!='verified input changed:protocol.json':raise
            results.append(dict(check='changed report input rejected',error=str(e)))
        else:raise AssertionError('changed reporting input accepted')
    (root/'verification_regression_checks.json').write_text(json.dumps(dict(status='passed',checks=results,
          checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--source-dir',required=True);a=p.parse_args();checks(a.root,a.source_dir)
