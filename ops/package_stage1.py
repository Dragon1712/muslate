"""Archive outputs plus the exact original checkpoints, without overwriting sources."""
from pathlib import Path
import argparse,json,hashlib,tarfile

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--checkpoint-root',required=True)
    ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.root).resolve();out=Path(a.out).resolve()
    assert not out.is_relative_to(root)
    assert json.loads((root/'final_integrity_audit.json').read_text())['status']=='passed'
    tests=json.loads((root/'final_remote_tests_verified.json').read_text())
    assert tests['total_tests']==140 and len(tests['files'])==20 and all(r['exit_code']==0 for r in tests['files'])
    assert len(list(root.glob('*/result.json')))==15 and len(list(root.glob('*/validation.json')))==27
    files={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()}
    assert all(not p.is_symlink() for p in files.values())
    original=json.loads((root/'original_checkpoint_sha256.json').read_text())
    for arm,sha in original.items():
        p=Path(a.checkpoint_root)/arm/'model.pt';assert digest(p)==sha
        name=f'{arm}/model.pt';assert name not in files;files[name]=p
    inventory={n:dict(sha256=digest(p),bytes=p.stat().st_size) for n,p in sorted(files.items())}
    manifest=root/'artifact_inventory.json'
    with manifest.open('x') as f:json.dump(dict(scope='all outputs and 27 original checkpoint bytes; excludes this manifest and outer archive',files=inventory),f,indent=2)
    files[manifest.name]=manifest
    with tarfile.open(out,'x:gz') as tar:
        for n,p in sorted(files.items()):tar.add(p,arcname=n,recursive=False)
    sha=digest(out)
    with out.with_suffix(out.suffix+'.sha256').open('x') as f:f.write(sha+'  '+out.name+'\n')
    print(json.dumps(dict(archive=str(out),sha256=sha,bytes=out.stat().st_size,n_files=len(files))))

if __name__=='__main__':main()
