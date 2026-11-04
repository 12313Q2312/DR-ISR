"""One-command entry point: verify invariants, then run the configured experiment."""

import sys
import unittest

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover("tests")
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    if "--test-only" not in sys.argv:
        from dr_isr.experiment import main
        main()
