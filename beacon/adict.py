import hashlib
import importlib.util
import json
import sys
import types

import yaml
import os
from collections import UserDict
from copy import deepcopy as dcp
from functools import wraps
from types import MappingProxyType
from typing import MutableMapping, Mapping, Callable

from beacon import xyz


# decorate internal methods in ADict
def mutate_attribute(fn):
    @wraps(fn)
    def decorator(*args, **kwargs):
        ctx = args[0]
        object.__setattr__(ctx, '_mutate_attribute', True)
        result = fn(*args, **kwargs)
        object.__setattr__(ctx, '_mutate_attribute', False)
        return result
    return decorator


class ADict(UserDict):
    @mutate_attribute
    def __init__(self, *args, **kwargs):
        if 'default' in kwargs:
            self._default = kwargs.pop('default')
            self._is_default_defined = True
        else:
            self._default = None
            self._is_default_defined = False
        mappings = dict()
        for mapping in args:
            if not isinstance(mapping, (dict, MutableMapping)):
                try:
                    mapping = dict(mapping)
                except (AttributeError, TypeError, ValueError):
                    raise TypeError(
                        f'Any of positional arguments must be able to converted to key-value type, '
                        f'but {mapping} is not.'
                    )
            mappings.update(mapping)
        self._frozen = False
        super().__init__(mappings, **kwargs)

    @property
    def frozen(self):
        return self._frozen

    def __getitem__(self, names):
        if isinstance(names, str):
            if names in self.data:
                value = self.data[names]
            elif self._is_default_defined:
                value = self.get_default()
                self.data[names] = value
            else:
                raise KeyError(f'The key "{names}" does not exist.')
        else:
            value = [self.__getitem__(name) for name in names]
        if self.frozen:
            value = dcp(value)
        return value

    def __setitem__(self, names, values):
        if not self.frozen:
            if isinstance(names, str):
                if isinstance(values, Mapping):
                    values = self.__class__(**values)
                elif isinstance(values, (list, tuple)):
                    values = [self.__class__(**value) if isinstance(value, Mapping) else value for value in values]
                super().__setitem__(names, values)
            elif isinstance(values, (list, tuple)):
                for name, value in zip(names, values):
                    self.__setitem__(name, value)
            else:
                for name in names:
                    self.__setitem__(name, values)

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            try:
                return self.__getitem__(name)
            except KeyError:
                raise AttributeError(name)

    def __setattr__(self, name, value):
        if self._mutate_attribute:
            object.__setattr__(self, name, value)
        else:
            self.__setitem__(name, value)

    def __delattr__(self, names):
        if self._mutate_attribute:
            object.__delattr__(self, names)
        elif isinstance(names, str):
            self.__delitem__(names)
        else:
            for name in names:
                self.__delitem__(name)

    def __deepcopy__(self, memo=None):
        mappings = dcp(self.data)
        kwargs = dict()
        if self._is_default_defined:
            kwargs.update(default=self._default)
        return self.__class__(mappings, **kwargs)

    def __getstate__(self):
        state = dcp(self.__dict__)
        state.pop('_mutate_attribute')
        return self.__dict__

    @mutate_attribute
    def __setstate__(self, state):
        self.data = state.pop('data')
        for k, v in state.items():
            object.__setattr__(self, k, v)

    def set_default(self, default=None):
        self._default = default

    def remove_default(self):
        self._is_default_defined = False
        self._default = None

    def get_default(self):
        if self._is_default_defined:
            _default = object.__getattribute__(self, '_default')
            if callable(_default):
                return _default()
            else:
                return dcp(_default)
        else:
            raise ValueError('Default value is not defined.')

    def get(self, name, default=None):
        if name in self:
            return self.__getitem__(name)
        elif self._is_default_defined:
            return self.get_default()
        else:
            return default

    def __delitem__(self, key):
        if not self.frozen:
            super().__delitem__(key)

    def pop(self, name, default=None):
        value = self.get(name, default)
        self.__delitem__(name)
        return value

    def raw(self, name, default=None):
        value = self.get(name, default)
        item = self.__class__(key=name, value=value)
        return item

    @mutate_attribute
    def filter(self, fn: Callable):
        data = dict()
        for key, value in self.items():
            if fn(key, value):
                data[key] = value
        self.data = data

    @mutate_attribute
    def freeze(self):
        self._frozen = True
        return self

    @mutate_attribute
    def defrost(self):
        self._frozen = False
        return self

    @mutate_attribute
    def update(self, __m=None, recurrent=False, **kwargs):
        if not self.frozen:
            if __m is not None:
                self.update(**__m, recurrent=recurrent)
            if recurrent:
                children = ADict()
                for k, v in kwargs.items():
                    if k in self and isinstance(v, Mapping):
                        self.__getitem__(k).update(**v, recurrent=True)
                    else:
                        children[k] = v
                super().update(**children)
            else:
                super().update(**kwargs)
        return self

    def get_structural_mapping(self, key, value):
        if key is None:
            key = ""
        items = []
        if isinstance(value, (MutableMapping, dict)):
            for k, v in value.items():
                concat = k if key == "" else f"{key}.{k}"
                items += self.get_structural_mapping(concat, v)
        else:
            return [(key, type(value).__name__)]
        return items

    def get_structural_repr(self):
        structural_repr = self.__class__()
        for k, v in self.get_structural_mapping("", self):
            structural_repr[k] = v
        return structural_repr

    def get_structural_hash(self):
        structural_repr = self.get_structural_repr()
        structural_repr = list(structural_repr.items())
        structural_repr.sort(key=lambda x: x[0])
        structural_hash = ''
        for k, v in structural_repr:
            structural_hash += f'[{k}:{v}]'
        return str(dcp(hashlib.sha1(structural_hash.encode('utf-8')).hexdigest()))

    @mutate_attribute
    def convert_to_immutable(self):
        self.data = MappingProxyType(self.data)

    @mutate_attribute
    def json(self):
        return json.dumps(self.to_dict())

    def clone(self):
        return dcp(self)

    def to_dict(self):
        data = dict()
        for key, value in self.items():
            if isinstance(value, Mapping):
                data[key] = self.__class__(**value).to_dict()
            elif isinstance(value, (list, tuple)):
                data[key] = [self.__class__(**item).to_dict() if isinstance(item, Mapping) else item for item in value]
            else:
                data[key] = value
        return data

    def to_xyz(self, format_dict=None):
        return xyz.dumps(self.to_dict(), format_dict=format_dict)

    @classmethod
    def from_file(cls, path):
        if os.path.exists(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.yml', '.yaml'):
                with open(path, 'rb') as f:
                    return cls(yaml.load(f, Loader=yaml.FullLoader))
            elif ext == '.json':
                with open(path, 'r') as f:
                    return cls(json.load(f))
            elif ext == '.xyz':
                return cls(xyz.load(path))
            elif ext == '.py':
                return cls(cls.from_python(path))
            else:
                raise ValueError(f'{ext} is not a valid file extension.')
        else:
            raise FileNotFoundError(f'{path} does not exist.')

    @classmethod
    def from_python(cls, path):
        config_name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(config_name, path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        sys.modules[config_name] = config_module
        config = {
            name: value
            for name, value in config_module.__dict__.items()
            if not name.startswith('__') and not isinstance(value, (types.ModuleType, types.FunctionType))
        }
        del sys.modules[config_name]
        return ADict(**config)

    def mm_like_update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, MutableMapping):
                recurrent = '_delete_' not in value
                if not recurrent:
                    del value['_delete_']
                if key in self and isinstance(self[key], MutableMapping) and recurrent:
                    self[key].mm_like_update(**value)
                else:
                    self[key] = value
            else:
                self[key] = value

    @classmethod
    def from_mm_config(cls, path):
        config = cls.from_file(path)
        mm_like_config = cls()
        if '_base_' in config:
            base_paths = config.pop('_base_')
            if isinstance(base_paths, str):
                base_paths = [base_paths]
            base_configs = [cls.from_mm_config(path) for path in base_paths]
        else:
            base_configs = []
        for base_config in base_configs:
            mm_like_config.mm_like_update(**base_config)
        mm_like_config.mm_like_update(**config)
        return mm_like_config

    @mutate_attribute
    def load(self, path):
        if os.path.exists(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.yml', '.yaml'):
                with open(path, 'rb') as f:
                    self.data = yaml.load(f, Loader=yaml.FullLoader)
            elif ext == '.json':
                with open(path, 'r') as f:
                    self.data = json.load(f)
            elif ext == '.xyz':
                self.data = xyz.load(path)
            elif ext == '.py':
                self.data = self.from_python(path).to_dict()
            else:
                raise ValueError(f'{ext} is not a valid file extension.')
        else:
            raise FileNotFoundError(f'{path} does not exist.')

    @mutate_attribute
    def load_mm_config(self, path):
        self.data = self.from_mm_config(path).to_dict()

    def dump(self, path):
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.yml', '.yaml'):
            with open(path, 'rb') as f:
                return yaml.dump(self.to_dict(), f, Dumper=yaml.Dumper)
        elif ext == '.json':
            with open(path, 'r') as f:
                return json.dump(self.to_dict(), f)
        elif ext == '.xyz':
            return xyz.dump(self, path)
        else:
            raise ValueError(f'{ext} is not a valid file extension.')

    def replace_keys(self, src_keys, tgt_keys):
        if len(src_keys) != len(tgt_keys):
            raise IndexError(f'Source and target keys cannot be mapped: {len(src_keys)} != {len(tgt_keys)}')
        for src_key in src_keys:
            if src_key not in self.data:
                raise KeyError(f'The key {src_key} does not exist.')
        self.__setitem__(tgt_keys, [self.data.pop(src_key) for src_key in src_keys])
