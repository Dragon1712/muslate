"""Run both unittest and function-test file entrypoints without losing tests."""
from pathlib import Path
import argparse,json,re,subprocess,sys,os

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--root');a=ap.parse_args()
    root=Path(a.root).resolve() if a.root else Path(__file__).resolve().parents[1];env=dict(os.environ)
    env['PYTHONPATH']=str(root)+os.pathsep+env.get('PYTHONPATH','')
    rows=[]
    test_files=sorted((root/'tests').glob('test_*.py'))
    if not test_files:raise ValueError(f'No test files under {root}/tests')
    for p in test_files:
        r=subprocess.run([sys.executable,str(p)],cwd=root,env=env,text=True,capture_output=True)
        output=r.stdout+'\n'+r.stderr
        m=re.search(r'(\d+) tests passed',output) or re.search(r'Ran (\d+) tests?',output)
        if not m:raise RuntimeError(f'No test count: {p.name}\n{output}')
        rows.append(dict(file=p.name,exit_code=r.returncode,n_tests=int(m[1]),stdout=r.stdout,stderr=r.stderr))
        print(f'{p.name}: {m[1]} tests, exit {r.returncode}',flush=True)
    report=dict(total_tests=sum(x['n_tests'] for x in rows),files=rows)
    with Path(a.out).open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    if any(x['exit_code'] for x in rows):raise SystemExit(1)
    print(f"PASS {report['total_tests']} tests in {len(rows)} files")

if __name__=='__main__':main()
