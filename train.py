"""Workspace-level wrapper for the SMACv2 training entry point.

Allows running from the repository root with:
  python -u train.py --condition 5 --seed 3
"""

from smacv2_poc.training.train import main


if __name__ == "__main__":
    main()
