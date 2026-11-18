"""Run from any directory with the project environment's Python."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import prepare

if __name__ == '__main__':
    prepare()
