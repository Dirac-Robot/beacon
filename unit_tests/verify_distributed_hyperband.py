import os

import torch

from beacon.adict import ADict
from beacon.hyperopt.hyperband import DistributedHyperBand
from beacon.scope import Scope
import torch.distributed as dist
import random

scope = Scope()
search_spaces = ADict(
    lr=ADict(param_type='FLOAT', param_range=(0.0001, 0.1), num_samples=10, space_type='LOG'),
    batch_size=ADict(param_type='INTEGER', param_range=(1, 64), num_samples=7,  space_type='LOG'),
    model_type=ADict(param_type='CATEGORY', categories=('resnet50', 'resnet101', 'swin_s'))
)
rank = int(os.environ.get('RANK', '0'))
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '1'))
torch.cuda.set_device(local_rank)
dist.init_process_group(
    backend='nccl',
    init_method='env:',
    rank=rank,
    world_size=world_size
)
rank = dist.get_rank()
world_size = dist.get_world_size()
hyperband = DistributedHyperBand(scope, search_spaces, 0.3, 4, None, mode='max', rank=rank, world_size=world_size)


@hyperband.main
def main(config):
    print(config.to_xyz())
    metric = random.random()
    return metric


if __name__ == '__main__':
    results = main()
    assert results.metric == max(results.logs[-1], key=lambda item: item.__metric__).__metric__


