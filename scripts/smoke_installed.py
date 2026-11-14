"""Exercise the installed package outside its source directory."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary)
    def call(*args):
        process=subprocess.run([sys.executable,'-I','-m','tabularmark_runner',*args],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if process.returncode:
            raise RuntimeError(process.stderr.decode('utf8',errors='replace'))
        return process.stdout.decode('utf8')
    subprocess.run([sys.executable,'-I','-m','tabularmark_runner.demo','--output','fixture'],cwd=root,check=True,stdout=subprocess.PIPE)
    call('--data-dir','fixture','--output-dir','run','--quick','--stages','rq1_detectability','rq1_roc','--device','cpu','--workers','1')
    call('--data-dir','fixture','--output-dir','run','--quick','--stages','rq1_detectability','rq1_roc','--device','cpu','--workers','1')
    report=json.loads((root/'run/manifest.json').read_text(encoding='utf8'))
    assert all(report['stages'][name]['status']=='completed' and report['stages'][name]['attempts']==1 for name in ['rq1_detectability','rq1_roc'])
    compare=[sys.executable,'-I','-m','tabularmark_ext.compare_cli','--input','fixture/cover_type_with_columns.csv','--output-dir','compare','--per-class','10','--n-key-cells','40','--watermark-length','10','--semmark-votes-per-bit','2','--alteration-percentages','0:100:50','--attack-repeats','1','--skip-ml']
    subprocess.run(compare,cwd=root,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (root/'notebooks').mkdir()
    (root/'notebooks/toy.ipynb').write_text(json.dumps({'cells':[{'cell_type':'code','source':["print('synthetic')"],'outputs':[],'execution_count':None}],'metadata':{},'nbformat':4,'nbformat_minor':5}),encoding='utf8')
    subprocess.run([sys.executable,'-I','-m','tabularmark_ext.audit_cli','--official-root','notebooks','--json-output','audit.json','--csv-output','audit.csv'],cwd=root,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert (root/'audit.csv').is_file()
print('PASS: installed CLI runs outside the source tree')
