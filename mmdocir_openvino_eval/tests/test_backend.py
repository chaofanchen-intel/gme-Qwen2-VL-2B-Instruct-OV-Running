import unittest

from mmdocir_eval.backend import create_backend


class BackendPluginTest(unittest.TestCase):
    def test_factory_receives_only_declared_arguments(self) -> None:
        backend = create_backend("fake_backend:create", value=3, ignored=4)

        self.assertEqual(backend.metadata(), {"value": 3})


if __name__ == "__main__":
    unittest.main()