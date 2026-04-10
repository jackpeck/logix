import unittest

import torch

from logix.analysis.influence_function_utils import precondition_kfac


class DummyState:
    def __init__(self, cov_eigval, cov_eigvec):
        self._cov_eigval = cov_eigval
        self._cov_eigvec = cov_eigvec

    def get_covariance_svd_state(self):
        return self._cov_eigval, self._cov_eigvec


class TestPreconditionKFAC(unittest.TestCase):
    def test_auto_damping_is_computed_per_module(self):
        src = {
            "module_a": {"grad": torch.tensor([[[1.0]]])},
            "module_b": {"grad": torch.tensor([[[1.0]]])},
        }
        cov_eigval = {
            "module_a": {
                "forward": torch.tensor([1.0]),
                "backward": torch.tensor([1.0]),
            },
            "module_b": {
                "forward": torch.tensor([3.0]),
                "backward": torch.tensor([1.0]),
            },
        }
        cov_eigvec = {
            "module_a": {
                "forward": torch.eye(1),
                "backward": torch.eye(1),
            },
            "module_b": {
                "forward": torch.eye(1),
                "backward": torch.eye(1),
            },
        }
        state = DummyState(cov_eigval, cov_eigvec)

        preconditioned = precondition_kfac(src=src, state=state, damping=None)

        self.assertTrue(
            torch.allclose(
                preconditioned["module_a"]["grad"],
                torch.tensor([[[1.0 / 1.1]]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                preconditioned["module_b"]["grad"],
                torch.tensor([[[1.0 / 3.3]]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
