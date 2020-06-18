from os.path import expanduser, join
from tqdm import tqdm

def pform(path, *names, sep= ''):
    """formats a path as `path` followed by `names` joined with `sep`."""
    return join(expanduser(path), sep.join(map(str, names)))

def save_txt(filename, lines, split=""):
    """writes lines to text file."""
    with open(filename, 'w') as file:
        for line in tqdm(lines):
            print(line+split, file= file)
