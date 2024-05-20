from pathlib import Path
from utils import *

class ConfigParser:
    def __init__(self,config):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        """
        # load config file and apply modification
        self.config = config

    @classmethod
    def from_args(cls,args):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        assert args.config is not None, 'Configuration needs specifying!'
        cfg_fname = Path(args.config)
        config = read_json(cfg_fname)
        return cls(config)

    def init_obj(self,name,module,*args,**kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args']) if 'args' in self[name] else dict()
        module_args.update(kwargs)
        return getattr(module,module_name)(*args,**module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]
