"""Run invariant checks then the full end-to-end neural comparison."""
import argparse
from datetime import datetime,timezone
import unittest

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/neural_benchmark.json")
    parser.add_argument("--data",default="neural_data")
    parser.add_argument("--output",default=None)
    parser.add_argument("--test-only",action="store_true")
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.discover("neural_tests")
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
    if not args.test_only:
        from neural.experiment import run
        run(args.config,args.data,args.output or "results/neural-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
