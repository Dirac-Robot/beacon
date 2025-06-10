import hashlib
import importlib.util
import json
import sys
import types
from collections.abc import MutableMapping as GenericMapping

import toml
import yaml
import os
from copy import deepcopy as dcp
from functools import wraps
from types import MappingProxyType
from typing import Mapping, MutableMapping, Callable

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


class Dict(GenericMapping):
    def __init__(self, mapping=None, /, **kwargs):
        self._data = dict()
        if mapping is not None:
            self.update(mapping)
        if kwargs:
            self.update(kwargs)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        if key in self._data:
            return self._data[key]
        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)

    def __setitem__(self, key, item):
        self._data[key] = item

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    # Modify __contains__ to work correctly when __missing__ is present
    def __contains__(self, key):
        return key in self._data

    # Now, add the methods in dicts but not in MutableMapping
    def __repr__(self):
        return repr(self._data)

    def __or__(self, other):
        if isinstance(other, Dict):
            return self.__class__(self._data | other._data)
        if isinstance(other, dict):
            return self.__class__(self._data | other)
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, Dict):
            return self.__class__(other._data | self._data)
        if isinstance(other, dict):
            return self.__class__(other | self._data)
        return NotImplemented

    def __ior__(self, other):
        if isinstance(other, Dict):
            self._data |= other._data
        else:
            self._data |= other
        return self

    def __copy__(self):
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        # Create a copy and avoid triggering descriptors
        inst.__dict__["_data"] = self.__dict__["_data"].copy()
        return inst

    def copy(self):
        if self.__class__ is Dict:
            return Dict(self._data.copy())
        import copy
        data = self._data
        try:
            self._data = dict()
            c = copy.copy(self)
        finally:
            self._data = data
        c.update(self)
        return c

    @classmethod
    def fromkeys(cls, iterable, value=None):
        d = cls()
        for key in iterable:
            d[key] = value
        return d


class ADict(Dict):
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
        self._accessed_keys = set()
        super().__init__(mappings, **kwargs)

    @property
    def frozen(self):
        return self._frozen

    @property
    def accessed_keys(self):
        return self._accessed_keys

    def get_minimal_config(self):
        new_config = ADict()
        for key in self.accessed_keys:
            value = self.__getitem__(key)
            if isinstance(value, self.__class__):
                new_config[key] = value.get_minimal_config()
            else:
                new_config[key] = value
        return new_config

    def __getitem__(self, names):
        if isinstance(names, str):
            if names in self._data:
                value = self._data[names]
            elif self._is_default_defined:
                value = self.get_default()
                self._data[names] = value
            else:
                raise KeyError(f'The key "{names}" does not exist.')
            self._accessed_keys.add(names)
        else:
            value = [self.__getitem__(name) for name in names]
            self._accessed_keys.update(names)
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
        mappings = dcp(self._data)
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
        self._data = state.pop('_data')
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
        if name in self:
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
        self._data = data

    def get_value_by_name(self, name):
        keys = name.split('.')
        value = self._data
        for key in keys:
            value = value[key]
        return value

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
                children = self.__class__()
                for k, v in kwargs.items():
                    if k in self and isinstance(v, Mapping):
                        self.__getitem__(k).update(**v, recurrent=True)
                    else:
                        children[k] = v
                super().update(**children)
            else:
                super().update(**kwargs)
        return self

    @mutate_attribute
    def update_if_absent(self, __m=None, recurrent=False, **kwargs):
        if not self.frozen:
            if __m is not None:
                self.update_if_absent(**__m, recurrent=recurrent)
            children = self.__class__()
            for k, v in kwargs.items():
                if k in self and isinstance(v, Mapping):
                    self.__getitem__(k).update_if_absent(**v, recurrent=True)
                elif k not in self:
                    children[k] = v
            super().update(**children)
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
        self._data = MappingProxyType(self._data)

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
            elif ext == '.toml':
                with open(path, 'r') as f:
                    return cls(toml.load(f))
            elif ext == '.json':
                with open(path, 'r') as f:
                    obj = json.load(f)
                    if isinstance(obj, list):
                        return [cls(item) for item in obj]
                    else:
                        return cls(obj)
            elif ext == '.jsonl':
                with open(path, 'r') as f:
                    dict_list = json.load(f)
                    return [cls(item) for item in dict_list]
            elif ext == '.xyz':
                obj = xyz.load(path)
                if isinstance(obj, list):
                    return [cls(item) for item in obj]
                else:
                    return cls(obj)
            elif ext == '.py':
                obj = cls.compile_from_file(path)
                if isinstance(obj, list):
                    return [cls(item) for item in obj]
                else:
                    return cls(obj)
            else:
                raise ValueError(f'{ext} is not a valid file extension.')
        else:
            raise FileNotFoundError(f'{path} does not exist.')

    @classmethod
    def compile_from_file(cls, path):
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
        return cls(**config)

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
            base_paths = [
                os.path.join(os.path.dirname(os.path.realpath(path)), base_path)
                if not os.path.exists(base_path) else base_path
                for base_path in base_paths
            ]
            base_configs = [cls.from_mm_config(path) for path in base_paths]
        else:
            base_configs = []
        for base_config in base_configs:
            mm_like_config.mm_like_update(**base_config)
        mm_like_config.mm_like_update(**config)
        return mm_like_config

    @mutate_attribute
    def load(self, path, **kwargs):
        if os.path.exists(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.yml', '.yaml'):
                with open(path, 'rb') as f:
                    self._data = yaml.load(f, Loader=yaml.FullLoader, **kwargs)
            elif ext == '.json':
                with open(path, 'r') as f:
                    self._data = json.load(f, **kwargs)
            elif ext == '.xyz':
                self._data = xyz.load(path)
            elif ext == '.py':
                self._data = self.compile_from_file(path).to_dict()
            else:
                raise ValueError(f'{ext} is not a valid file extension.')
        else:
            raise FileNotFoundError(f'{path} does not exist.')

    @mutate_attribute
    def load_mm_config(self, path):
        self.update(self.from_mm_config(path).to_dict())

    def dump(self, path, **kwargs):
        dir_path = os.path.dirname(os.path.realpath(path))
        os.makedirs(dir_path, exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.yml', '.yaml'):
            with open(path, 'w') as f:
                return yaml.dump(self.to_dict(), f, Dumper=yaml.Dumper, **kwargs)
        elif ext == 'toml':
            with open(path, 'w') as f:
                return toml.dump(self.to_dict(), f, **kwargs)
        elif ext == '.json':
            with open(path, 'w') as f:
                return json.dump(self.to_dict(), f, **kwargs)
        elif ext == '.xyz':
            return xyz.dump(self.to_dict(), path, **kwargs)
        else:
            raise ValueError(f'{ext} is not a valid file extension.')

    def replace_keys(self, src_keys, tgt_keys):
        if len(src_keys) != len(tgt_keys):
            raise IndexError(f'Source and target keys cannot be mapped: {len(src_keys)} != {len(tgt_keys)}')
        for src_key in src_keys:
            if src_key not in self._data:
                raise KeyError(f'The key {src_key} does not exist.')
        self.__setitem__(tgt_keys, [self._data.pop(src_key) for src_key in src_keys])


