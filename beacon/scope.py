import argparse
import inspect
import sys
import uuid
import warnings
from contextlib import contextmanager
from functools import wraps

from beacon.adict import ADict
from inspect import currentframe, getframeinfo

from beacon.parser import parse_command


# safe compile
def exec_with_no_permissions(code, __locals):
    exec(compile(code, '<string>', 'single'), {'__builtins__': None}, __locals)


def parse_args_pythonic():
    if Scope.stored_arguments is not None:
        _pythonic_vars = Scope.stored_arguments
    else:
        _pythonic_vars = sys.argv[1:]
    joined = ' '.join(_pythonic_vars)
    _pythonic_vars = parse_command(joined)
    pythonic_vars = []
    for literal in _pythonic_vars:
        if '=' not in literal:
            pythonic_vars.extend(literal.split())
        else:
            name, *values = literal.split('=')
            value = '='.join(values)
            if value.startswith('%') and value.endswith('%'):
                value = f'"{value[1:-1]}"'.replace('%', '\"')
            pythonic_vars.append(f'{name}={value}')
    default_prefix = ''
    if len(Scope.registry) == 1:
        scope_name = list(Scope.registry.keys())[0]
        default_prefix += f'{scope_name}.'
    for literal in pythonic_vars:
        literal = default_prefix+literal
        scope_name = literal.split('.')[0]
        scope = Scope.registry.get(scope_name)
        if scope is not None:
            literal = literal[len(f'{scope_name}.'):]
            if literal in scope.views or '=' in literal:
                scope.assign(literal)
    Scope.parsed = True


def _add_func_to_scope(scope, func, field=None, priority=0, lazy=False, default=False, chain_with=None):
    field = field or func.__name__
    chain_with = chain_with or []
    if isinstance(chain_with, str):
        chain_with = [chain_with]
    scope.views[field] = {
        "view_type": 'function',
        "priority": priority,
        "lazy": lazy,
        "fn": func,
        "chain_with": chain_with,
        "default": default
    }
    if default:
        scope.assign(field)


def add_func_to_scope(scope, field=None, priority=0, lazy=False, default=False, chain_with=None):
    def decorator(func):
        _add_func_to_scope(scope, func, field, priority, lazy, default, chain_with)
        return func
    return decorator


def add_config_to_scope(scope, field=None, config=None, priority=0, lazy=False, default=False, chain_with=None):
    if field is None:
        raise ValueError(f'A name of config must be specified to assign config directly.')
    config = config or ADict()
    view = ADict()
    view.view_type = 'config'
    view.config = config
    view.priority = priority
    view.lazy = lazy
    chain_with = chain_with or []
    if isinstance(chain_with, str):
        chain_with = [chain_with]
    view.chain_with = chain_with
    view.default = default
    scope.views[field] = view
    if default:
        scope.assign(field)


def add_func_to_multi_scope(scopes, field=None, priority=0, lazy=False, default=False, chain_with=None):
    def decorator(func):
        for scope in scopes:
            _add_func_to_scope(scope, func, field, priority, lazy, default, chain_with)
        return func
    return decorator


def add_config_to_multi_scope(scopes, field=None, config=None, priority=0, lazy=False, default=False, chain_with=None):
    for scope in scopes:
        add_config_to_scope(scope, field, config, priority, lazy, default, chain_with)


def patch_parsing_method(func, unknown_external_literals='error'):
    modes = {'merge', 'ignore', 'error'}
    if unknown_external_literals not in {'merge', 'ignore', 'error'}:
        raise ValueError(
            f'Unexpected unknown_external_literals="{unknown_external_literals}"; '
            f'Choose the one from {modes}.'
        )

    @wraps(func)
    def capture(*args, **kwargs):
        parser = args[0]
        if len(sys.argv) >= 2:
            if sys.argv[1] in ('--help', '-h'):
                warnings.warn(f'Scope does not support "--help" option. Use "manual" instead.', DeprecationWarning)
            if sys.argv[1] in ('--help', '-h', 'manual'):
                print('[External Parser]')
                parser.print_help()
                Scope.logging_manual()
                sys.exit(0)
        known, unknown = func(*args, **kwargs)
        defaults = vars(func(args[0], [])[0])
        non_defaults = ADict(vars(known)).clone()
        non_defaults.filter(lambda k, v: k not in defaults or v != defaults[k])
        stored_arguments = []
        while unknown:
            literal = unknown.pop(0)
            if literal.startswith('--'):
                if unknown_external_literals == 'error':
                    raise RuntimeError(f'Unexpected external literal: {literal}')
                value = unknown.pop(0)
                key = literal[2:].replace('-', '_')
                if unknown_external_literals == 'merge':
                    exec_with_no_permissions(
                        compile(f'non_defaults["{key}"] = {value}', '<string>', 'single'),
                        __locals={'non_defaults': non_defaults}
                    )
            else:
                stored_arguments.append(literal)
        Scope.stored_arguments = stored_arguments
        for scope in Scope.registry.values():
            if scope.use_external_parser:
                scope.observe('_argparse', defaults, scope.external_priority, lazy=False)
                scope.assign('_argparse')
                scope.observe('_argparse_literals', non_defaults, scope.external_priority, lazy=True)
                scope.assign('_argparse_literals')
        parse_args_pythonic()
        release()
        return known, unknown

    return capture


def release():
    if hasattr(_parser, 'stored_methods'):
        _parser.parse_args, _parser.parse_known_args = _parser.stored_methods
        del _parser.stored_methods


_parser = argparse.ArgumentParser


def _print_config(config):
    print(config.to_xyz())
    sys.exit(0)


class Scope:
    registry = ADict()
    parsed = False
    stored_arguments = None
    current_scope = None
    unknown_external_literals = 'ignore'

    def __init__(
        self,
        config=None,
        name='config',
        use_external_parser=False,
        external_priority=-2,
        enable_override=False
    ):
        self.config = ADict() if config is None else config
        self.name = name
        self.use_external_parser = use_external_parser
        self.enable_override = enable_override
        self.register()
        self.views = ADict()
        self.manuals = ADict()
        self.observe('_default', config, priority=-1, lazy=False)
        add_func_to_scope(self, 'print', priority=1280, lazy=True, default=False)(_print_config)
        self.screen = ADict(views=[], literals=[], lazy_views=[])
        self.external_priority = external_priority
        self.compute = False
        self.config_in_compute = None
        self.mode = 'ON'
        self.is_applied = False

    def activate(self):
        self.mode = 'ON'

    def deactivate(self):
        self.mode = 'OFF'

    @contextmanager
    def pause(self):
        self.deactivate()
        yield
        self.activate()
    
    def register(self):
        registry = self.__class__.registry
        if len(registry) == 0:
            _parser.stored_methods = [_parser.parse_args, _parser.parse_known_args]
            _parser.parse_args = _parser.parse_known_args = patch_parsing_method(
                _parser.parse_known_args,
                self.__class__.unknown_external_literals
            )
        if self.name in registry and not self.enable_override:
            raise ValueError(
                f'{self.name} is already used by another scope. '
                f'The name of scope must not be duplicated.'
            )
        self.__class__.registry[self.name] = self

    @classmethod
    def initialize_registry(cls):
        cls.registry.clear()

    @classmethod
    def override(cls, scope):
        cls.registry[scope.name] = scope

    def add_to_screen(self, field=None, config=None, priority=0, lazy=False, default=False, chain_with=None):
        if config is not None:
            add_config_to_scope(self, field, config, priority, lazy, default, chain_with)
        else:
            return add_func_to_scope(self, field, priority, lazy, default, chain_with)

    def observe(self, field=None, config=None, priority=0, lazy=False, default=False, chain_with=None):
        # to enable replace from external codes
        return self.__class__.registry[self.name].add_to_screen(field, config, priority, lazy, default, chain_with)

    def manual(self, field=None, manual=None):
        if manual is not None:
            self.manuals.update(manual)
        else:
            field(self.manuals)

    @classmethod
    def logging_manual(cls):
        for scope_name, scope in cls.registry.items():
            print('-'*50)
            print(f'[Scope "{scope_name}"]')
            print('(The Applying Order of Views)')
            view_names = list([key for key, view in scope.views.items() if not view.lazy])
            lazy_view_names = list([key for key, view in scope.views.items() if view.lazy])
            view_names.sort(key=lambda x: scope.views[x].priority)
            lazy_view_names.sort(key=lambda x: scope.views[x].priority)
            print(' → '.join(view_names+['(CLI Inputs)']+lazy_view_names))
            print('(User Manuals)')
            manuals = scope.manuals
            if len(cls.registry) > 1:
                manuals = manuals.clone()
                for key, value in scope.manuals.items():
                    manuals[f'{scope.name}.{key}'] = manuals.pop(key)
            print(manuals.to_xyz())
        print('-'*50)
        sys.exit(0)

    def get_assigned_views(self):
        default_views = [view for view in self.screen.views if self.views[view].default]
        views = [view for view in self.screen.views if view not in default_views]
        default_lazy_views = [lazy_view for lazy_view in self.screen.lazy_views if self.views[lazy_view].default]
        lazy_views = [lazy_view for lazy_view in self.screen.lazy_views if lazy_view not in default_lazy_views]
        return ADict(
            default_views=default_views,
            default_lazy_views=default_lazy_views,
            views=views,
            lazy_views=lazy_views,
            literals=self.screen.literals
        )

    def assign(self, literals):
        if not isinstance(literals, (list, tuple)) or isinstance(literals, str):
            literals = [literals]
        for literal in literals:
            if literal in self.views:
                view_info = self.views[literal]
                if view_info.lazy and literal not in self.screen.lazy_views:
                    if view_info.chain_with is not None:
                        self.assign(view_info.chain_with)
                    self.screen.lazy_views.append(literal)
                elif not view_info.lazy and literal not in self.screen.views:
                    if view_info.chain_with is not None:
                        self.assign(view_info.chain_with)
                    self.screen.views.append(literal)
            else:
                self.screen.literals.append(literal)

    def apply(self):
        if len(sys.argv) >= 2:
            if sys.argv[1] in ('--help', '-h'):
                warnings.warn(f'Scope does not support "--help" option. Use "manual" instead.', DeprecationWarning)
            if sys.argv[1] in ('--help', '-h', 'manual'):
                Scope.logging_manual()
        self.__class__.current_scope = self
        self.screen.views.sort(key=lambda x: self.views[x].priority)
        self.screen.lazy_views.sort(key=lambda x: self.views[x].priority)
        for field in self.screen.views:
            view = self.views[field]
            if view.view_type == 'config':
                self.config.update(view.config)
            else:
                view.fn(self.config)
        for literal in self.screen.literals:
            if isinstance(literal, str):
                exec_with_no_permissions(f'config.{literal}', __locals={'config': self.config})
            else:
                self.config.update(literal)
        self.compute = True
        self.config.freeze()
        for field in self.screen.views:
            view = self.views[field]
            if view.view_type != 'config':
                view.fn(self.config)
        self.compute = False
        self.config.defrost()
        for field in self.screen.lazy_views:
            view = self.views[field]
            if view.view_type == 'config':
                self.config.update(view.config)
            else:
                view.fn(self.config)
        self.is_applied = True

    def __enter__(self):
        if not Scope.parsed:
            parse_args_pythonic()
        if not self.is_applied:
            self.apply()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _recreate_context(self):
        return self

    def _is_config_at_positional(self, func, *args, **kwargs):
        if self.name in kwargs:
            return True
        full_arguments = inspect.getfullargspec(func)[0]
        index = full_arguments.index(self.name)
        return {'positional': index+1 < len(args) or len(full_arguments) > len(args), 'index': index}

    def get_config_updated_arguments(self, func, *args, **kwargs):
        pos_info = self._is_config_at_positional(func, *args, **kwargs)
        if pos_info['positional']:
            args = list(args)
            args.insert(pos_info['index'], self.config)
        else:
            kwargs.update({self.name: self.config})
        return args, kwargs

    def exec(self, func):
        @wraps(func)
        def inner(*args, **kwargs):
            with self._recreate_context():
                if self.mode == 'ON':
                    args, kwargs = self.get_config_updated_arguments(func, *args, **kwargs)
                return func(*args, **kwargs)
        return inner

    def __call__(self, func):
        return self.exec(func)

    def reset_user_inputs(self):
        self.screen = ADict(views=[], literals=[], lazy_views=[])

    def convert_argparse_to_scope(self):
        args = self.views['_argparse'].config
        code = f"def argparse({self.name}):\n"
        code += '\n'.join([
            f'    {self.name}.{key} = '+(f"'{value}'" if isinstance(value, str) else f'{value}')
            for key, value in args.items()
        ])
        return code

    @classmethod
    @contextmanager
    def lazy(cls, with_compile=False, priority=1):
        scope = cls.current_scope
        if scope.compute and not with_compile:
            scope.config.defrost()
            yield
            scope.config.freeze()
        else:
            scope.config.freeze()
            try:
                yield
            except (KeyError, AttributeError):
                pass
            scope.config.defrost()
        if with_compile and not scope.compute:
            frame = currentframe().f_back.f_back
            frame_info = getframeinfo(frame)
            file_name = frame_info.filename
            start = frame_info.positions.lineno
            end = frame_info.positions.end_lineno
            with open(file_name, 'r') as f:
                inner_ctx_lines = list(f.readlines())[start:end]
            ctx_name = f"_lazy_context_{str(uuid.uuid4()).replace('-', '_')}"
            inner_ctx_lines = [f'def {ctx_name}({scope.name}):']+inner_ctx_lines
            global_vars = frame.f_globals
            local_vars = frame.f_locals
            exec(compile('\n'.join(inner_ctx_lines), '<string>', 'exec'), global_vars, local_vars)
            scope.observe(default=True, lazy=True, priority=priority)(local_vars[ctx_name])


class MultiScope:
    def __init__(self, *scopes):
        self.scopes = scopes
        self.register_all()
        if Scope.parsed:
            for scope in self.scopes:
                scope.reset_user_inputs()
            parse_args_pythonic()

    def register_all(self):
        Scope.registry.clear()
        for scope in self.scopes:
            Scope.registry[scope.name] = scope

    def __call__(self, func):
        def decorator(*args, **kwargs):
            arguments = inspect.getfullargspec(func)
            name_spaces = set(arguments.args+arguments.kwonlyargs)
            for scope in self.scopes:
                if scope.name in name_spaces or arguments.varkw is not None:
                    args, kwargs = scope.get_config_updated_arguments(func, *args, **kwargs)
                scope.apply()
            return func(*args, **kwargs)
        return decorator
