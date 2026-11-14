import contextlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from tabularmark_runner.demo import generate
from tabularmark_runner.runner import _run_lock,build_parser
from tabularmark_ext.audit_cli import main as audit_main


class PublicCLITests(unittest.TestCase):
    def test_defaults_are_cpu_and_offline(self):
        args=build_parser().parse_args([])
        self.assertEqual(args.device,'cpu');self.assertFalse(args.allow_downloads)
        self.assertEqual(args.data_dir,Path('private/data'))

    def test_concurrent_output_lock_and_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            with _run_lock(directory):
                with self.assertRaises(RuntimeError):
                    with _run_lock(directory):pass
            with _run_lock(directory):pass

    def test_comparison_cli_preserves_secret_state_and_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);generate(root/'fixture')
            command=[sys.executable,'-m','tabularmark_ext.compare_cli','--input',str(root/'fixture/cover_type_with_columns.csv'),
                     '--output-dir',str(root/'result'),'--per-class','10','--n-key-cells','40','--watermark-length','10',
                     '--semmark-votes-per-bit','2','--alteration-percentages','0:100:50','--attack-repeats','1','--skip-ml']
            result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr.decode())
            states=list((root/'result').glob('watermark_state_*.json'))
            self.assertEqual(len(states),3)
            for state in states:
                self.assertTrue(json.loads(state.read_text()))
                if os.name!='nt':self.assertEqual(stat.S_IMODE(state.stat().st_mode),0o600)
            repeated=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
            self.assertNotEqual(repeated.returncode,0)

    def test_audit_cannot_write_inside_its_source_tree(self):
        with tempfile.TemporaryDirectory() as temporary,contextlib.redirect_stderr(io.StringIO()):
            root=Path(temporary);source=root/'source';source.mkdir()
            argv=['audit','--official-root',str(source),'--json-output',str(source/'out.json'),'--csv-output',str(root/'out.csv')]
            with patch.object(sys,'argv',argv),self.assertRaises(SystemExit) as error:audit_main()
            self.assertEqual(error.exception.code,2)
            self.assertEqual(list(source.iterdir()),[])


if __name__=='__main__':unittest.main()
