"""Download della versione 3 del dataset (~2.70 GB dichiarati, spazio extra per estrazione)."""
import argparse
from pathlib import Path
import kagglehub
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/mushrooms');a=p.parse_args()
    root=kagglehub.dataset_download('tinkhoav/mushroom-classification/versions/3',output_dir=str(Path(a.output).resolve()))
    print('Dataset:',root)
    print('Per il training usa --data con la cartella images che contiene le sottocartelle delle classi.')
