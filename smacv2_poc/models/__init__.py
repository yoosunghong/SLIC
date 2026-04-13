"""
Re-export policy builders from validation_poc.
All 5 conditions share the same model code — only the environment changes.
Condition 6 (MAPPO Vanilla flat obs) is defined locally in baselines/.
"""
import os
import sys

# Ensure repo-level packages are importable when this package is imported directly.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import importlib
_vp_models = importlib.import_module('validation_poc.models') if 'validation_poc.models' not in __import__('sys').modules else __import__('sys').modules['validation_poc.models']

from validation_poc.models import build_policies, CentralizedCritic  # noqa: F401
from validation_poc.models.base_policy import BasePolicy              # noqa: F401
from validation_poc.models.full_model import FullModel                # noqa: F401
from validation_poc.models.no_self_attn import NoSelfAttnModel        # noqa: F401
from validation_poc.models.role_mlp import RoleMLPModel               # noqa: F401
from validation_poc.models.no_role_label import NoRoleLabelModel      # noqa: F401
from validation_poc.models.flat_mlp import FlatMLPModel               # noqa: F401
