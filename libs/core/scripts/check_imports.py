"""Script to check if python modules can be imported."""

import random
import string
import sys
import traceback
from importlib.machinery import SourceFileLoader

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        has_failure = True
        print(file)
        traceback.print_exc()
        print()

    sys.exit(1 if has_failure else 0)
