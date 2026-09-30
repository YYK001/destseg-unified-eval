import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from zipfile import ZipFile
from external_baselines.destseg_visa.processes import run_process


class ProcessTests(unittest.TestCase):
    def invoke(self, source, timeout=5):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            path=Path(d)/'child.log'
            event=threading.Event()
            run_process([sys.executable,'-u','-c',source],path,'test',event,
                        idle_timeout=timeout,heartbeat=.1)
            return path.read_text()

    def test_success_preserves_output_and_exit(self):
        log=self.invoke("print('checkpoint saved',flush=True)")
        self.assertIn('checkpoint saved',log)
        self.assertIn('exited code=0',log)

    def test_failed_child_does_not_look_complete(self):
        with self.assertRaisesRegex(RuntimeError,'exit code 3'):
            self.invoke('raise SystemExit(3)')

    def test_heartbeat_does_not_hide_stalled_child(self):
        start=time.monotonic()
        with self.assertRaises(TimeoutError):
            self.invoke('import time; time.sleep(30)',timeout=.5)
        self.assertLess(time.monotonic()-start,8)

    def test_cancel_stops_other_process(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            event=threading.Event(); event.set()
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                run_process([sys.executable,'-u','-c','import time; time.sleep(30)'],
                            Path(d)/'child.log','test',event)

    def test_wall_deadline_stops_even_with_output(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(TimeoutError,'deadline'):
                run_process([sys.executable,'-u','-c',
                    "import time\nwhile True:\n print('working',flush=True)\n time.sleep(.05)"],
                    Path(d)/'child.log','test',threading.Event(),idle_timeout=10,
                    deadline=time.monotonic()+.5)

    def test_budget_does_not_start_unfinishable_category(self):
        import json
        from external_baselines.destseg_visa.pipeline import main, can_start_category
        self.assertFalse(can_start_category(10,20,15))
        self.assertTrue(can_start_category(10,None,15))
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); out=root/'eval'
            argv=['pipeline','--phase','all','--dataset-root',d,'--dtd-root',d,
                  '--training-dir',str(root/'train'),'--output-dir',str(out),
                  '--categories','candle','--max-hours','.1']
            with patch.object(sys,'argv',argv), patch('external_baselines.destseg_visa.pipeline.run_process') as run:
                main()
                run.assert_not_called()
            state=json.loads((root/'eval_logs/pipeline_status.json').read_text())
            self.assertEqual(state['status'],'partial_budget')
            self.assertEqual(state['completed'],[])
            self.assertEqual(state['pending'],['candle'])

    def test_failure_archives_records_and_existing_final_weight(self):
        from external_baselines.destseg_visa.pipeline import main
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); train=root/'train'; out=root/'eval'
            weight=train/'candle/train/DeSTSeg_VISA_5000_candle.pckl'
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b'synthetic-test-not-a-model')
            argv=['pipeline','--phase','all','--dataset-root',str(root),'--dtd-root',str(root),
                  '--training-dir',str(train),'--output-dir',str(out),'--categories','candle']
            with patch.object(sys,'argv',argv), patch(
                'external_baselines.destseg_visa.pipeline.run_process',side_effect=TimeoutError('test stall')):
                with self.assertRaises(TimeoutError): main()
            with ZipFile(str(out)+'_partial_records.zip') as z:
                self.assertIn('eval_logs/pipeline_failure.json',z.namelist())
                self.assertEqual(json.loads(z.read('eval_logs/pipeline_status.json'))['status'],'failed')
            with ZipFile(str(out)+'_recovery.zip') as z:
                self.assertIn('train/candle/train/DeSTSeg_VISA_5000_candle.pckl',z.namelist())


if __name__=='__main__':
    unittest.main()
