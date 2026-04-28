"""
Miracle 1.0.1 - Pytest Configuration
=====================================
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
