"""Re-export AstaClient for compatibility with src/tools/science_tools.py.

Import from src.core.agents.asta directly in all new code.
"""
from src.core.agents.asta import AstaClient

__all__ = ["AstaClient"]
