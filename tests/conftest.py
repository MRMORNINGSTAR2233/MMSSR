"""
Tests configuration
"""
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Disable telemetry for tests
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
