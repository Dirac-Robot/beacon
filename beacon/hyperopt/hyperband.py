import math
import uuid
from itertools import product, chain

import numpy as np

from beacon.adict import ADict
import torch.distributed as dist

from copy import deepcopy as dcp


class HyperOpt:
    def __init__(self, scope, search_spaces, tracker=None, mode='max'):
        if mode not in ('min', 'max'):
            raise ValueError('mode must be either "min" or "max".')
        self.scope = scope
        self.search_spaces = search_spaces
        self.config = scope.config.clone()
        self.tracker = tracker
        self.mode = mode
        self.config.__hyperopt_id__ = self.get_hyperopt_id()

    @classmethod
    def get_hyperopt_id(cls):
        return str(uuid.uuid4())

    def main(self, func):
        raise NotImplementedError()


class DistributedMixIn:
    def __init__(self, rank=0, world_size=1, backend='pytorch'):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend

    @property
    def is_root(self):
        return self.rank == 0

    def broadcast_object_from_root(self, obj):
        if self.backend == 'pytorch':
            obj = [obj]
            dist.broadcast_object_list(obj)
            obj = obj[0]
        elif self.backend == 'beacon':
            raise NotImplementedError()  # todo
        else:
            raise ValueError(f'Unsupported backend: {self.backend}')
        return obj

    def all_gather_object(self, obj):
        if self.backend == 'pytorch':
            gathered_objects = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_objects, obj)
        elif self.backend == 'beacon':
            raise NotImplementedError()  # todo
        else:
            raise ValueError(f'Unsupported backend: {self.backend}')
        return gathered_objects

    def destroy(self):
        if self.backend == 'pytorch':
            if dist.is_initialized():
                dist.destroy_process_group()
        elif self.backend == 'beacon':
            raise NotImplementedError()  # todo
        else:
            raise ValueError(f'Unsupported backend: {self.backend}')

    def get_hyperopt_id(self):
        return self.broadcast_object_from_root(str(uuid.uuid4()))


class DistributedHyperOpt(DistributedMixIn, HyperOpt):
    def __init__(
        self,
        scope,
        search_spaces,
        tracker=None,
        mode='max',
        rank=0,
        world_size=1,
        backend='pytorch'
    ):
        HyperOpt.__init__(self, scope, search_spaces, tracker, mode)
        DistributedMixIn.__init__(self, rank, world_size, backend)


class GridSpaceMixIn:
    @classmethod
    def prepare_distributions(cls, base_config, search_spaces):
        sampling_spaces = ADict()
        for param_name, search_space in search_spaces.items():
            if 'param_type' not in search_space:
                raise KeyError(f'param_type for parameter {param_name} is not defined at search_spaces.')
            param_type = search_space['param_type'].upper()
            if param_type == 'INTEGER':
                start, stop = search_space.param_range
                space_type = search_space.get('space_type', 'LINEAR')
                if space_type == 'LINEAR':
                    optim_space = np.linspace(
                        start=start,
                        stop=stop,
                        num=search_space.num_samples,
                        dtype=np.int64
                    ).tolist()
                elif space_type == 'LOG':
                    base = search_space.get('base', 2)
                    optim_space = np.logspace(
                        start=math.log(start, base),
                        stop=math.log(stop, base),
                        num=search_space.num_samples,
                        dtype=np.int64,
                        base=base
                    ).tolist()
                else:
                    raise ValueError(f'Invalid space_type: {space_type}')
            elif param_type == 'FLOAT':
                start, stop = search_space.param_range
                space_type = search_space.get('space_type', 'LINEAR')
                if space_type == 'LINEAR':
                    optim_space = np.linspace(
                        start=start,
                        stop=stop,
                        num=search_space.num_samples,
                        dtype=np.float32
                    ).tolist()
                elif space_type == 'LOG':
                    base = search_space.get('base', 10)
                    optim_space = np.logspace(
                        start=math.log(start, base),
                        stop=math.log(stop, base),
                        num=search_space.num_samples,
                        dtype=np.float32,
                        base=base
                    ).tolist()
                else:
                    raise ValueError(f'Invalid space_type: {space_type}')
            elif param_type == 'CATEGORY':
                optim_space = search_space.categories
            else:
                raise ValueError(f'Unknown param_type for parameter {param_name}; {param_type}')
            sampling_spaces[param_name] = optim_space
        grid_space = [ADict(zip(sampling_spaces.keys(), values)) for values in product(*sampling_spaces.values())]
        distributions = [
            dcp(base_config).update(**partial_config, __num_halved__=0)
            for index, partial_config in enumerate(grid_space)
        ]
        return distributions


class HyperBand(HyperOpt, GridSpaceMixIn):
    def __init__(self, scope, search_spaces, halving_rate, num_min_samples, tracker=None, mode='max'):
        if halving_rate <= 0 or halving_rate >= 1:
            raise ValueError(f'halving_rate must be greater than 0.0 but less than 1.0, but got {halving_rate}.')
        if num_min_samples < 1:
            raise ValueError(f'num_min_samples must be greater than or equal to 1, but got {num_min_samples}.')
        super().__init__(scope, search_spaces, tracker, mode)
        self.halving_rate = halving_rate
        self.num_min_samples = num_min_samples
        self.distributions = self.prepare_distributions(self.config, self.search_spaces)

    def main(self, func):
        def launch(*args, **kwargs):
            logs = []
            distributions = self.distributions
            while len(distributions) >= self.num_min_samples:
                results = self.estimate(func, distributions, *args, **kwargs)
                results.sort(key=lambda item: item.__metric__, reverse=self.mode == 'max')
                logs.append(results)
                distributions = []
                for config in results[:int(len(results)*self.halving_rate)]:
                    config.__num_halved__ += 1
                    distributions.append(config)
            last_config = logs[-1][0]
            metric = self.estimate_single_run(func, last_config, *args, **kwargs)
            best_config = dcp(last_config)
            best_config.__metric__ = metric
            logs.append([best_config])
            return ADict(config=best_config, metric=metric, logs=logs)
        return launch

    def estimate(self, estimator, distributions, *args, **kwargs):
        results = []
        for config in distributions:
            config = dcp(config)
            self.scope.config = config
            metric = self.estimate_single_run(estimator, config, *args, **kwargs)
            config.__metric__ = metric
            results.append(config)
        return results

    def estimate_single_run(self, estimator, config, *args, **kwargs):
        self.scope.config = config
        return self.scope(estimator)(*args, **kwargs)

    def num_generations(self):
        max_size = len(self.distributions)
        min_size = self.num_min_samples
        return math.ceil(math.log(min_size/max_size)/math.log(self.halving_rate))

    def compute_optimized_initial_training_steps(self, max_steps):
        num_generations = self.num_generations()
        min_steps = max_steps*math.pow(self.halving_rate, num_generations)
        return [
            *(math.ceil(min_steps/math.pow(self.halving_rate, index)) for index in range(1, num_generations)),
            max_steps
        ]


class DistributedHyperBand(DistributedMixIn, HyperBand, GridSpaceMixIn):
    def __init__(
        self,
        scope,
        search_spaces,
        halving_rate,
        num_min_samples,
        tracker=None,
        mode='max',
        rank=0,
        world_size=1,
        backend='pytorch'
    ):
        DistributedMixIn.__init__(self, rank, world_size, backend)
        HyperBand.__init__(self, scope, search_spaces, halving_rate, num_min_samples, tracker, mode)

    def estimate(self, estimator, distributions, *args, **kwargs):
        batch_size = math.ceil(len(distributions)/self.world_size)
        distributions = distributions[self.rank*batch_size:(self.rank+1)*batch_size]
        results = super().estimate(estimator, distributions, *args, **kwargs)
        results = list(chain(*self.all_gather_object(results)))
        return results

    def estimate_single_run(self, estimator, config, *args, **kwargs):
        result = super().estimate(estimator, config, *args, **kwargs)
        self.destroy()
        return result


