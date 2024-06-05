import uuid
from itertools import product

import numpy as np

from beacon.adict import ADict


class HyperOpt:
    def __init__(self, scope, space_config, tracker, mode='max'):
        self.scope = scope
        self.enabled_params = set(scope.config.keys())
        self.space_config = space_config
        self.config = scope.config.clone()
        self.tracker = tracker
        self.mode = mode
        self.add_hyperopt_id()

    def add_hyperopt_id(self):
        self.config.__hyperopt_id__ = str(uuid.uuid4())


class DistributedHyperOpt(HyperOpt):
    def __init__(self, scope, space_config, tracker, backend='pytorch', mode='max'):
        super().__init__(scope, space_config, tracker, mode)
        self.init_distributed(backend)

    def init_distributed(self, backend):


class HyperBand(HyperOpt):
    def __init__(self, scope, space_config, tracker, mode='max'):
        super().__init__(scope, space_config, tracker, mode)
        self.distributions = []
        self.superiors = []
        self.inferiors = []
        self.init_distributions()

    def init_distributions(self):
        sampling_spaces = ADict()
        for param_name, space_info in self.space_config.items():
            if param_name not in self.enabled_params:
                raise KeyError(f'Parameter {param_name} is not defined at scope.')
            if 'param_type' not in space_info:
                raise KeyError(f'param_type for parameter {param_name} is not defined at space_config.')
            param_type = space_info['param_type'].upper()
            if param_type == 'INTEGER':
                start, end = space_info.param_range
                optim_space = list(range(start, end, space_info.interval))
            elif param_type == 'FLOAT':
                start, end = space_info.param_range
                optim_space = np.linspace(start, end, space_info.max_num_values)
            elif param_type == 'CATEGORY':
                optim_space = space_info.categories
            else:
                raise ValueError(f'Unknown param_type for parameter {param_name}; {param_type}')
            sampling_spaces[param_name] = optim_space
        grid_space = [ADict(zip(sampling_spaces.keys(), values)) for values in product(*sampling_spaces.values())]
        self.distributions = [
            ADict(**self.config).update(**partial_config)
            for index, partial_config in enumerate(grid_space)
        ]

    def submit(self, config, metric):
        config.__metric__ = metric
        self.tracker.write()

