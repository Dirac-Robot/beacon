# Beacon
Beacon is collections of useful tools to manage various experiments easily.

### ADict
`ADict` is variation of `dict` type and used to configure experiments in `Scope`. It supports all methods in `dict` and
#### Some Useful Features Compared to Python Native Dictionary
- You can access items as attributes.
```python
from beacon.adict import ADict

config = ADict(lr=0.1, optimizer='SGD')


if __name__ == '__main__':
    print(config.lr == config['lr'])
```
The results will be:
```shell
True
```
- You can update items only if they do not exist:
```python
from beacon.adict import ADict

config = ADict(lr=0.1, optimizer='SGD')

if __name__ == '__main__':
    config.update_if_absent(lr=0.01, scheduler='CosineAnnealingLR')
    print(config.lr == 0.01)  # False
    print(config.scheduler == 'CosineAnnealingLR')  # True
```
- You can save or load directly, and it will automatically handle file extensions.
- Supported file formats are: `json`, `yaml`, `yml`, `toml`, `xyz`.
```python
from beacon.adict import ADict

if __name__ == '__main__':
    config = ADict().load('config.json')  # or ADict.from_file('config.json')
    config.dump('config.json')
    # Alternatively, you can use any other supported file extension.
```
### Scope
Configurations can bother experiments when it is not carefully managed. Scope decreases potentials of mistakes on 
setup experiments or reproducing previous results.

#### Assign Custom Configs to Scope
```python
# run.py

from beacon.scope import Scope

scope = Scope()


@scope.observe()
def my_config(config):
    config.alpha = 0  # FYI, config['alpha'] can also be used


@scope
def main(config):
    print(config.alpha)


if __name__ == '__main__':
    main()
```

`@scope.observe()` is simple python decorator to define presets of configs. Configs can be combined into scope, and
config with higher priority will be merged later. And, if priorities of configs are same, they will be collated 
in order.

```python
# run.py

from beacon.scope import Scope

scope = Scope()


@scope.observe()  # priority == 0 if it is not specified
def normal_config(config):
    config.alpha = 0


@scope.observe(priority=1)
def important_config(config):
    config.alpha = 1


@scope.observe(priority=2)
def most_important_config(config):
    config.alpha = 2


@scope
def main(config):
    print(config.alpha)


if __name__ == '__main__':
    main()
```

Then observed configurations will be collated reversed order of priorities.

```shell
python train.py normal_config  # 0
python train.py normal_config important_config  # 1
python train.py normal_config important_config most_important_config  # 2
```

Customized configs or python literals can be defined via CLI environment also.

```shell
python train.py config_1 config_2 lr=0.01 model.backbone.type=%ResNet50% model.backbone.embed_dims=128 model.backbone.depths=[2, 2, 18, 2]
```

For convenience, python literals defined via CLI always have the highest priorities. 
Do not forget to wrap strings by "%" instead of quotes, or error will be thrown. 
(There are few special characters that bash allows.)

---

#### Feed Configs to Runtime Functions
To feed `config` to runtime functions, you can use `@scope` decorator. Note that it can wrap any function you want to 
feed `config`. Or, you can get config from scope directly by two-line codes:

```python
from beacon.scope import Scope

scope = Scope()

# ...some customized configs

if __name__ == '__main__':
    scope.apply()
    config = scope.config
    
    # ...write your codes
```

If you want to use config without wrapping functions with `@scope`, you must execute `scope.apply()` first. It applies 
all configs to scope manually, including configs and python literals declared from CLI.

Priorities enforce the applying order of configurations, and you can prevent your experiments from mistakes come from
complicated configuration structure.

---

#### Define Type of Configs
When configs are assigned to scope, you can specify several options to customize: `name`, `default` and `lazy`.

When configs are wrapped by `@scope.observe(default=True)`, configs will be applied as default.

```python
# run.py

from beacon.scope import Scope

scope = Scope()


@scope.observe(default=True)
def my_default_config(config):
    config.alpha = 0

    
@scope.observe()
def my_config(config):
    config.alpha = 0.1


@scope
def main(config):
    print(config.alpha)

    
if __name__ == '__main__':
    main()
```

```shell
# CLI

python run.py  # 0
python run.py my_config  # 0.1
```

It will be useful when some parameters are frequently used while you do not want to fix them manually.

When you want to let some configs define computed properties via logic blocks, configs with `lazy=True` will help:

```python
# run.py

from beacon.scope import Scope

scope = Scope()


@scope.observe(default=True)
def my_default_config(config):
    config.alpha = 0
    config.beta = 1
    if config.alpha > 0:  # It does not have any effect, because alpha is always defined as 0 before this block
        config.beta = 0.1


@scope.observe(lazy=True)
def my_lazy_config(config):
    # Priorities of lazy configs are always higher than that of python literals.
    # So, logic blocks can check CLI commands and edit beta dynamically.
    if config.alpha > 0:  
        config.beta = 0.1


@scope
def main(config):
    print(config.beta)

    
if __name__ == '__main__':
    main()
```

```shell
# CLI

python run.py  # 0
python run.py alpha=1  # 1
python run.py alpha=1 my_lazy_config  # 0.2
```

From `python>=3.11`, you can make logic blocks behave like lazy configs via context blocks:

```python
# run.py

from beacon.scope import Scope

scope = Scope()


@scope.observe()
def my_config(config):
    config.alpha = 1
    config.beta = 1
    with Scope.lazy():  # It will be computed after CLI commands are assigned, even alpha is already defined above.
        if config.alpha > 1:
            config.beta = 2


@scope
def main(config):
    print(config.beta)

    
if __name__ == '__main__':
    main()
```

```shell
python run.py my_config alpha=2  # 2
```

It enables logical code blocks, like conditional stuffs, can be computed considering customized user inputs via 
CLI commands. Note that codes in `Scope.lazy()` context blocks affect to config only, unless when `with_compile` is 
specified as `True`. Although `with_compile` uses safe compile, you can make it disable if considering 
unexpected accidents. 

---

### Advanced Usages
Decorating your functions with `Scope` is very convenient, but it may harm to implement clean codes. For instance,
if some people does not want to use existing `Scope`, but their own `Scope` instead,
pre-implemented decorators frustrate them. But the main motto of `beacon` is 
"Do not bother essential things with trivial things". So, you can easily override `Scope` or use multiple `Scope`'s in your codes. 

#### Manual Control
`Scope` can be turned on or off manually, and even it is very simple.
```python
scope = ...
scope.activate()  # scope is activated
scope.deactivate()  # scope is deactivated

with scope.pause():
    ...  # works without scope
```

#### Override
If existing codes are wrapped with another scope, it can be overridden using `Scope.override`. It is class-method to
assign new scope suppressing error if its name is already defined in global registry.

```python
from beacon.scope import Scope

scope = Scope(name='existing_config')  # existing scope

my_scope = Scope(name='existing_config')
Scope.override(my_scope)  # put your scope
```

Even existing `Scope` is overridden, all observed configs are preserved and can be assigned to new `Scope` instance.

---

#### MultiScope
If multiple scopes should be used at same code, they can be wrapped easily by defining them as `MultiScope`.

```python
from beacon.scope import Scope, MultiScope


scope_1 = Scope(name='config_1')
scope_2 = Scope(name='config_2')

unified_scope = MultiScope(scope_1, scope_2)


@unified_scope
def main(config_1, config_2):
    ...  # Do something
```

Configs in `MultiScope` will be identified by their names in `@unified_scope` for this example. Note that 
`@unified_scope.observe` is NOT supported, because it makes the order of priorities complicated much. 
Of course, you still can use `@scope_1.observe` and `@scope_2.observe` independently.

---

#### Import and Export from External System
Various types of configs can be imported from `json`, `yaml`, `py` and `xyz` formats and exported to 
`json`, `yaml`, `py`, and `xyz` formats. 

`xyz` is customized file extension for our config and supports more readable formatting.

Also, config can be imported from OpenMMLab configurations, but it is experimental feature yet.

```python
scope = ...

@scope.observe()
def import_from_config(config):
    config.load('/path/to/your/config.any_supported_ext')  # import from external config
    config.dump('/path/to/your/config.any_supported_ext')
    config.load_mm_config('/path/to/your/OpenMMLab/config.py')  # import from external OpenMMLab config
```

---

#### Merge with `argparse`
Scope supports merging `argparse` to `Scope`. The only you have to do is set `use_external_parser=True`. 

```python
from beacon.scope import Scope
import argparse


scope = Scope(use_external_parser=True)
parser = argparse.ArgumentParser()
...  # Do something


if __name__ == '__main__':
    parser.parse_args()
```

Then `parser.parse_args()` let your arguments merged to your scope. And, of course, it can be mixed with original usage 
of `Scope`.


#### Write Manual
Many configuration libraries including `argparse` supports printing manuals for each parameter. `@scope.manual` is easy 
but strong method to write manual. It is almost same with `@scope.observe` but you can put manuals in config instead of 
values.

If manual is declared in any `Scope`, including `MuitiScope`, then it is displayed when `manual` argument is specified
to command line.

```python
from beacon.scope import Scope, MultiScope

scope_1 = Scope(name='config_1')
scope_2 = Scope(name='config_2')
scope = MultiScope(scope_1, scope_2)


@scope_1.observe(default=True)
def default_1(config_1):
    config_1.lr = 0.1


@scope_1.manual
def default_manual_1(config_1):
    config_1.lr = 'learning rate.'


@scope_2.observe(default=True)
def default_2(config_2):
    config_2.batch_size = 16


@scope_2.manual
def default_manual_2(config_2):
    config_2.batch_size = 'batch size.'


@scope
def main(config_1, config_2):
    pass


if __name__ == '__main__':
    main()
```

Then its manual output will be:

```shell
$ python main.py manual
[Scope "config_1"]
config_1.lr: learning rate.
[Scope "config_2"]
config_2.batch_size: batch size.
```

Probably you do not want to drop manuals out declared in `argparse`. And do not worry about it, it will be printed too,
same way as `argparse`.

```python
from beacon.scope import Scope
import argparse

scope = Scope(use_external_parser=True)
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=16, type=int, help='batch size.')


@scope.observe(default=True)
def default(config):
    config.lr = 0.1


@scope.manual
def default_manual(config):
    config.lr = 'learning rate.'


@scope
def main(config):
    pass


if __name__ == '__main__':
    parser.parse_args()
    main()
```

Then its manual output is:

```shell
$ python main.py manual
[External Parser]
usage: verify_argparse_manual.py [-h] [--batch-size BATCH_SIZE]

options:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        batch size.
[Scope "config"]
lr: learning rate.
```

Internally, `argparse` is treated as another scope, so note that its parameters are detached from that of your `scope`'s
when manuals are printed out.

#### Use Structural Hash to Classify Experiment Settings
`get_structural_hash()` method in `ADict` makes hashed results of **structure** of configuration. If the keys and 
the **types** of values are same, their structural hashes are same too. Note that `float` type is compatible with `int`
type, but `int` type is not compatible with `float` type.

```python
from beacon.adict import ADict

a1 = ADict(lr=0.1, rank=0, optimizer='AdamW') 
a2 = ADict(lr=0.2, rank=1, optimizer='SGD') 
b1 = ADict(lr=0.1, rank='0', optimizer='AdamW')
b2 = ADict(lr=0.2, rank='1', optimizer='SGD')


if __name__ == '__main__':
    print(a1.get_structural_hash() == a2.get_structural_hash())  # True
    print(a1.get_structural_hash() == b1.get_structural_hash())  # False
    print(b1.get_structural_hash() == b2.get_structural_hash())  # True
```

The results will be:
```shell
True
False
True
```

This feature will be useful when you want to find if the setup of experiment is different with previous or not 
automatically.

### Experimental Features
#### Hyperparameter Optimization via Hyperband
It is experimental feature.

With `Scope`, you can use hyperparameter optimization algorithms. `HyperBand` is class for feed hyperparameters 
to search via scope.

Hyperband algorithm uses successive halving algorithms and early-stopping mechanisms. As `Scope` cannot adjust your
experiments directly, it feeds hidden parameter, `__num_halved__`, to your config. You can use this to adjust the number
of steps to apply early-stopping.

```python
from beacon.adict import ADict
from beacon.hyperopt.hyperband import HyperBand
from beacon.scope import Scope

scope = Scope()
# define configurations of hyperparameters to search
search_spaces = ADict(
    lr=ADict(param_type='FLOAT', param_range=(0.0001, 0.1), num_samples=10, space_type='LOG'),
    batch_size=ADict(param_type='INTEGER', param_range=(1, 64), num_samples=7,  space_type='LOG'),
    model_type=ADict(param_type='CATEGORY', categories=('resnet50', 'resnet101', 'swin_s'))
)
# halving_rate means surviving rate from successors
# If the number of next experiments decreases under num_min_samples, it is terminated and return config with best result
# mode determines how to sort metrics
hyperband = HyperBand(scope, search_spaces, halving_rate=0.3, num_min_samples=4, mode='max')


@hyperband.main
def main(config):
    metric = ...  # some metric that you want to optimize
    return metric


if __name__ == '__main__':
    results = main()
    print(results.config)
```

If you do not want to determine the number of steps to early-stopping manually, 
`compute_optimized_initial_training_steps` method may be helpful. It computes the number of steps for each generation
of Hyperband algorithm automatically.

```python
from beacon.adict import ADict
from beacon.hyperopt.hyperband import HyperBand
from beacon.scope import Scope

scope = Scope()
search_spaces = ADict(
    lr=ADict(param_type='FLOAT', param_range=(0.0001, 0.1), num_samples=10, space_type='LOG'),
    batch_size=ADict(param_type='INTEGER', param_range=(1, 64), num_samples=7,  space_type='LOG'),
    model_type=ADict(param_type='CATEGORY', categories=('resnet50', 'resnet101', 'swin_s'))
)
# halving_rate means surviving rate from successors
# If the number of next experiments decreases under num_min_samples, it is terminated and return config with maximum
# mode determines how to sort metrics
hyperband = HyperBand(scope, search_spaces, halving_rate=0.3, num_min_samples=4, mode='max')
max_steps = 120000
print(hyperband.compute_optimized_initial_training_steps(max_steps))  # [27, 88, 292, 972, 3240, 10800, 36000, 120000]
```

In the future, distributed hyperband and some other HyperOpt algorithms will be added soon.