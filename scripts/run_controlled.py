"""Prepare, smoke-test, run, and summarize controlled v2 experiments."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.controlled_data import ROOT, prepare_controlled
from src.controlled_training import run_grid


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/controlled_v2.json')
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'main', 'controls', 'summarize', 'all'], required=True)
    parser.add_argument('--policies', nargs='+', choices=['include_sensitive', 'exclude_sensitive'])
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.stage in ['prepare', 'all']:
        prepare_controlled(config)
    for stage in (['smoke', 'main', 'controls'] if args.stage == 'all' else [args.stage]):
        if stage in ['smoke', 'main', 'controls']:
            run_grid(config, stage, args.policies, args.seeds)
    if args.stage in ['summarize', 'all']:
        from src.controlled_analysis import summarize_controlled
        summarize_controlled(config)
