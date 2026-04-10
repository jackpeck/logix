import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from logix.batch_info import BatchInfo
from logix.logging.log_saver import LogSaver
from logix.logging.mmap import MemoryMapHandler
from logix.state import LogIXState


class TestLogSaverAsync(unittest.TestCase):
    def _make_saver(self, log_dir):
        config = SimpleNamespace(log_dir=log_dir, num_workers=2, flush_threshold=1)
        state = LogIXState()
        state.set_state("model_module", path=[["dummy_data"]], path_dim=[2])
        return LogSaver(config=config, state=state)

    def _make_batch_info(self):
        binfo = BatchInfo()
        binfo.data_id = ["item-1"]
        binfo.log["dummy_data"] = torch.tensor([[1.0, 2.0]])
        return binfo

    def test_async_flush_writes_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = self._make_saver(tmpdir)
            saver.buffer_write(self._make_batch_info())

            saver.flush()
            saver.finalize()

            self.assertTrue(
                os.path.exists(os.path.join(tmpdir, "log_rank_0_chunk_0.mmap"))
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(tmpdir, "log_rank_0_chunk_0_metadata.json")
                )
            )
            self.assertEqual(saver._pending_flushes, [])

    def test_async_flush_surfaces_worker_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = self._make_saver(tmpdir)
            saver.buffer_write(self._make_batch_info())

            with mock.patch.object(
                MemoryMapHandler, "write", side_effect=RuntimeError("boom")
            ):
                saver.flush()
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    saver.finalize()


if __name__ == "__main__":
    unittest.main()
