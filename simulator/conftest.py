"""Make the flat simulator modules importable under pytest.

The v0.2 package ships as flat top-level modules (``adaface_embedding_simulator``,
``faceaudit_adapter``, ...) imported without a package prefix, mirroring the upstream
standalone layout. Inserting this directory on ``sys.path`` lets those imports resolve
whether the tests are run from the repo root or from inside ``simulator/``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
