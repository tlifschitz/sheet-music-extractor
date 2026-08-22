import sys
from pathlib import Path

# The scripts live at the repo root rather than in an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
