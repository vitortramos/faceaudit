"""Make the flat simulator modules importable under pytest.

The package uses flat top-level modules (``adaface_embedding_simulator``,
``faceaudit_seed``, ...) imported without a package prefix. Inserting this directory on
``sys.path`` lets those imports resolve whether the tests are run from the repo root or
from inside ``simulator/``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
