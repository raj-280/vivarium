# conftest.py — shared pytest configuration
import sys
from pathlib import Path

# Make the vivarium package importable during tests
sys.path.insert(0, str(Path(__file__).parent.parent))
