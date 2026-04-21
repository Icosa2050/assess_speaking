import logging
import sys
import tempfile
import unittest

from app_backend.config import build_backend_runtime_config
from scripts import run_backend


class RunBackendLoggingTests(unittest.TestCase):
    def test_configure_backend_logging_writes_to_rotating_log_file(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as cache_dir:
                config = build_backend_runtime_config(
                    app_data_dir=app_dir,
                    cache_dir=cache_dir,
                    port=8764,
                )
                run_backend._configure_backend_logging(config.log_file)
                sys.stdout.write("backend hello\n")
                sys.stdout.flush()
                logging.getLogger("uvicorn.error").warning("uvicorn warning")
                sys.stderr.write("backend error\n")
                sys.stderr.flush()
                contents = config.log_file.read_text(encoding="utf-8")

            self.assertIn("backend hello", contents)
            self.assertIn("uvicorn warning", contents)
            self.assertIn("backend error", contents)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    unittest.main()
