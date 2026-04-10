import tempfile
import unittest

import torch

from logix.state import LogIXState


class TestLogIXState(unittest.TestCase):
    def test_covariance_inverse_recomputes_when_damping_changes(self):
        covariance = torch.diag(torch.tensor([1.0, 2.0]))

        state = LogIXState()
        state.covariance_state["layer"]["grad"] = covariance

        expected_inverse_1 = torch.diag(torch.tensor([1.0 / 2.0, 1.0 / 3.0]))
        inverse_1 = state.get_covariance_inverse_state(damping=1.0)
        self.assertTrue(
            torch.allclose(inverse_1["layer"]["grad"], expected_inverse_1)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            state.save_state(tmpdir)

            restored_state = LogIXState()
            restored_state.load_state(tmpdir)

            restored_inverse_1 = restored_state.get_covariance_inverse_state(
                damping=1.0
            )
            self.assertTrue(
                torch.allclose(
                    restored_inverse_1["layer"]["grad"],
                    expected_inverse_1,
                )
            )

            expected_inverse_2 = torch.diag(torch.tensor([1.0 / 3.0, 1.0 / 4.0]))
            restored_inverse_2 = restored_state.get_covariance_inverse_state(
                damping=2.0
            )
            self.assertTrue(
                torch.allclose(
                    restored_inverse_2["layer"]["grad"],
                    expected_inverse_2,
                )
            )


if __name__ == "__main__":
    unittest.main()
