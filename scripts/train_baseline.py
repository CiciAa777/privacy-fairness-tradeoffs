import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.baseline import train

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Train ordinary SGD employment MLP')
    parser.add_argument('--config',type=Path,default=None)
    train(parser.parse_args().config)
