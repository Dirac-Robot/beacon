import uuid
import torch.distributed as dist


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
        else:
            raise ValueError(f'Unsupported backend: {self.backend}')
        return obj

    def all_gather_object(self, obj):
        if self.backend == 'pytorch':
            gathered_objects = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_objects, obj)
        else:
            raise ValueError(f'Unsupported backend: {self.backend}')
        return gathered_objects

    def destroy(self):
        if self.backend == 'pytorch':
            if dist.is_initialized():
                dist.destroy_process_group()
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
