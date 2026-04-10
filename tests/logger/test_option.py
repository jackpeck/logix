import unittest

from logix.logging.option import LogOption


class TestLogOption(unittest.TestCase):
    def test_setup_rejects_unknown_keys(self):
        option = LogOption()

        with self.assertRaisesRegex(
            ValueError,
            "Unknown logging option keys: \\['log', 'save', 'statistic'\\]",
        ):
            option.setup({"log": "grad", "save": "grad", "statistic": "kfac"})


if __name__ == "__main__":
    unittest.main()
