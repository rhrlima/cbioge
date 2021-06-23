import glob

from cbioge.analyze import plots
from cbioge.utils import checkpoint as ckpt

if __name__ == '__main__':
    
    files = glob.glob('small/10484/data_*.ckpt')
    files.sort(key=lambda f: ckpt.natural_key(f))
    print(files)

    
