### Scope Tutorial
Scope is tool to manage configurations for various experiments easily.

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

`scope.observe()` is simple python decorator to define presets of configs. Configs can be combined into scope, and
config with higher priority will be merged later. And, if priorities of configs are same,

Customized configs or python literals can be defined via CLI environment also.

```shell
python train.py config_1 config_2 lr=0.01 model.backbone.type=\'ResNet50\' model.backbone.embed_dims=128 model.backbone.depths=[2, 2, 18, 2]
```

For convenience, python literals defined via CLI always have highest priorities. Do not forget to wrap strings by quotes 
with escape(\' or \"), or bash automatically removes all quotes from arguments.

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
python run.py my_config  # alpha == 0.1
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

---

#### Import and Export from External System
Various types of configs can be imported from `json`, `yaml`, `py` and `xyz` formats and exported to 
`json`, `yaml`, `py`, and `xyz` formats. 

`xyz` is customized file system for our config and supports more readable formatting.

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
