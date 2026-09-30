import contextlib
import io
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
                self.assertNotIn('complete',z.read('eval_logs/pipeline_status.json').decode())
            with ZipFile(str(out)+'_recovery.zip') as z:
                self.assertIn('train/candle/train/DeSTSeg_VISA_5000_candle.pckl',z.namelist())


if __name__=='__main__':
    unittest.main()
