import io
import subprocess
import sys
import unittest
from threading import Thread

from metaflow_extensions.torchrun.plugins.torchrun_libs.streaming import (
    stream_subprocess_output,
)


class StreamSubprocessOutputTest(unittest.TestCase):
    def run_child(self, script):
        with subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            stdout = io.StringIO()
            stderr = io.StringIO()
            outcome = {}

            def stream_output():
                try:
                    outcome["returncode"] = stream_subprocess_output(
                        process, stdout, stderr
                    )
                except BaseException as error:
                    outcome["error"] = error

            thread = Thread(target=stream_output)
            thread.start()
            thread.join(timeout=5)

            if thread.is_alive():
                process.kill()
                thread.join(timeout=5)
                self.fail("subprocess output draining deadlocked")

            if "error" in outcome:
                raise outcome["error"]

        return outcome["returncode"], stdout.getvalue(), stderr.getvalue()

    def test_large_stdout_does_not_wait_for_stderr(self):
        payload_size = 1024 * 1024
        returncode, stdout, stderr = self.run_child(
            "import sys; sys.stdout.write('x' * %d); sys.stdout.flush()"
            % payload_size
        )

        self.assertEqual(0, returncode)
        self.assertEqual(payload_size, len(stdout))
        self.assertEqual("", stderr)

    def test_large_stderr_does_not_wait_for_stdout(self):
        payload_size = 1024 * 1024
        returncode, stdout, stderr = self.run_child(
            "import sys; sys.stderr.write('x' * %d); sys.stderr.flush()"
            % payload_size
        )

        self.assertEqual(0, returncode)
        self.assertEqual("", stdout)
        self.assertEqual(payload_size, len(stderr))

    def test_drains_both_streams_through_process_exit(self):
        returncode, stdout, stderr = self.run_child(
            "import sys; "
            "sys.stdout.write('final stdout'); "
            "sys.stderr.write('final stderr'); "
            "raise SystemExit(7)"
        )

        self.assertEqual(7, returncode)
        self.assertEqual("final stdout", stdout)
        self.assertEqual("final stderr", stderr)


if __name__ == "__main__":
    unittest.main()
