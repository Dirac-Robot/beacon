import numpy as np


class HyperBand:
    def __init__(self, scope, space_config, tracker):
        self.scope = scope
        self.enabled_params = set(scope.config.keys())
        self.space_config = space_config
        self.configs = [scope.config.clone()]
        self.tracker = tracker

    def set_space(self, partial_configs):
        for index in range(len(self.configs)):
            config = self.configs[index]
            self.configs.extend([config.clone().update(partial_config) for partial_config in partial_configs])

    def generate_space(self):
        for param_name, space_info in self.space_config.items():
            if param_name not in self.enabled_params:
                raise KeyError(f'Parameter {param_name} is not defined at scope.')
            if 'param_type' not in space_info:
                raise KeyError(f'param_type for parameter {param_name} is not defined at space_config.')
            param_type = space_info['param_type']
            if param_type.upper() == 'INTEGER':
                start, end = space_info.param_range
                optim_space = list(range(start, end, space_info.interval))
            elif param_type.upper() == 'FLOAT':
                start, end = space_info.param_range
                optim_space = np.linspace(start, end, space_info.max_num_values)
            elif param_type.upper() == 'CATEGORY':
                optim_space = space_info.categories
            else:
                raise ValueError(f'Unknown param_type for parameter {param_name}; {param_type.upper()}')
            self.set_space([{param_name: value} for value in optim_space])
