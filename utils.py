import torch
import numpy as np
from pathlib import Path
import json
from collections import OrderedDict

def random_seed_setup(seed:int=None):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if seed:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        print(f'Random seed set to be: {seed}')

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle,object_hook=OrderedDict)
