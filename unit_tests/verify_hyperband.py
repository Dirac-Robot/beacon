from beacon.adict import ADict
from beacon.hyperopt.hyperband import HyperBand
from beacon.scope import Scope
import random

scope = Scope()
search_spaces = ADict(
    lr=ADict(param_type='FLOAT', param_range=(0.0001, 0.1), num_samples=8, space_type='LOG'),
    batch_size=ADict(param_type='INTEGER', param_range=(1, 64), num_samples=10,  space_type='LOG'),
    model_type=ADict(param_type='CATEGORY', categories=('resnet50', 'resnet101', 'swin_s', 'vit-s'))
)
hyperband = HyperBand(scope, search_spaces, 0.3, 4, None)


@hyperband.main
def main(config):
    print(config.to_xyz())
    metric = random.random()
    return metric


if __name__ == '__main__':
    results = main()
    print(hyperband.num_generations())
    print(hyperband.compute_optimized_initial_training_steps(24))
    assert results.metric == max(results.logs[-1], key=lambda item: item.__metric__).__metric__


