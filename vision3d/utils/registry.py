"""Registry for managing various components in the Vision3D framework."""
from vision3d.utils.logging import configure_logger

logger = configure_logger(__name__.split(".")[-1])

class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register(self, cls=None, name=None):
        if cls is None:
            def decorator(cls):
                self._register(cls, name or cls.__name__)
                return cls
            return decorator
        self._register(cls, name or cls.__name__)
        return cls

    def _register(self, cls, name):
        if name in self._module_dict:
            raise KeyError(f'{name} is already registered in {self._name}')
        self._module_dict[name] = cls

    def get(self, name):
        return self._module_dict.get(name)

    def build(self, cfg):
        assert isinstance(cfg, dict) and 'type' in cfg
        cls = self.get(cfg['type'])
        if cls is None:
            raise KeyError(f"{cfg['type']} is not registered in {self._name}")
        logger.info(f"Building {cls.__name__} from {self._name} registry.")
        return cls(**{k: v for k, v in cfg.items() if k != 'type'})
    

# Define registries for different components
MODELS = Registry('models')
DATASETS = Registry('datasets')
LOSSES = Registry('losses')
HOOKS = Registry('hooks')
METRICS = Registry('metrics')
UTILS = Registry('utils')