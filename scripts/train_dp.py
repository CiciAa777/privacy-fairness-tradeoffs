import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dp import run_pilots

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Run Opacus DP-SGD pilots')
    parser.add_argument('--config',type=Path,default=None)
    parser.add_argument('--epsilons',type=float,nargs='+',default=None)
    args=parser.parse_args()
    run_pilots(args.config,args.epsilons)
