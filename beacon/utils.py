from typing import Iterable, Union, Sequence, Optional, Dict


def is_seq(x):
    return isinstance(x, Iterable) and not isinstance(x, str)


def get_all(iterable, key):
    return map(lambda x: x[key], iterable)


def _to_ord(s):
    if s is not None and len(s) >= 1:
        return ord(s)


def replace_all(
    string: str,
    sources: Union[Sequence[str], str],
    mappings: Optional[Union[Dict[str, str], Sequence[str], str]] = None
):
    if mappings is None:
        mappings = {}
    elif not isinstance(mappings, dict):
        assert len(sources) == len(mappings), 'sources and mappings must have same length.'
        mappings = {source: target for source, target in zip(sources, mappings)}
    mappings = {ord(source): _to_ord(mappings.get(source, None)) for source in sources}
    return string.translate(mappings)


def remove_all(string, targets):
    for target in targets:
        string = string.replace(target, '')
    return string


def convert_string_to_value(value):
    if not isinstance(value, str):
        return value
    if value.lower() == 'none':
        return None
    elif value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value == '[Empty Sequence]':
        return []
    elif value == '[Empty Mapping]':
        return dict()
    elif value.isdecimal():
        return int(value)
    elif remove_all(value.lower(), ('.', 'e+', 'e-')).isnumeric():
        return float(value)
    else:
        return value
