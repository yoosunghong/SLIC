"""
Re-export visualisation functions from validation_poc.
All plots are environment-agnostic — no changes needed.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from validation_poc.analysis.visualize import (  # noqa: F401
    plot_self_attn_heatmap,
    plot_role_routing_matrix,
    plot_delta_max_comparison,
    plot_gradient_attribution,
    plot_win_rate_curves,
)
