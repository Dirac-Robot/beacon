import os
from typing import Iterable

from beacon.utils import convert_string_to_value


class GlobalParser:
    def __init__(self, value, value_type, parent=None, format_dict=None):
        self.value = int(value) if value_type == 'index' else value
        self.node_type = 'value'  # initialized to value
        self.value_type = value_type  # Type of the node (either 'key', 'index', or 'value')
        self.parent = parent  # Parent node
        self.children = []  # List to hold children nodes
        self.format_dict = format_dict

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self
        if child_node.value_type == 'key':
            self.node_type = 'dict'
        elif child_node.value_type == 'index':
            self.node_type = 'iter'
        else:
            self.node_type = 'item'

    def dumps(self, level=None):
        decoding_format_str = ''
        if self.format_dict:
            if self.parent is None:
                for key, value in self.format_dict.items():
                    decoding_format_str += f'{key} -> {value}'
                    decoding_format_str += '\n'
            prefix = self.format_dict.get(f'{self.value_type}_prefix', '')
            postfix = self.format_dict.get(f'{self.value_type}_postfix', '')
            if self.value_type == 'key':
                postfix = postfix or ':'
            elif self.value_type == 'index':
                postfix = postfix or ')'
        else:
            prefix = ''
            if self.value_type == 'key':
                postfix = ':'
            elif self.value_type == 'index':
                postfix = ')'
            else:
                postfix = ''
        xyz_raw_str = ''
        if self.parent:
            xyz_raw_str += '  '*level+prefix+str(self.value)+postfix
        else:
            level = -1
        if not self.children:
            if self.node_type == 'iter':
                xyz_raw_str += ' [Empty Sequence]'
            elif self.node_type == 'dict':
                xyz_raw_str += ' [Empty Mapping]'
        for child in self.children:
            if child.value_type != 'value':
                xyz_raw_str += '\n'+child.dumps(level+1)
            else:
                child_value = f'"{child.value}"' if isinstance(child.value, str) else str(child.value)
                xyz_raw_str += ' '+child_value
        xyz_raw_str = decoding_format_str+xyz_raw_str
        # If decoding format is empty, remove first line break
        if xyz_raw_str and xyz_raw_str[0] == '\n':
            xyz_raw_str = xyz_raw_str[1:]
        return xyz_raw_str

    def __repr__(self):
        return f'Node({", ".join([f"{name}={value}" for name, value in vars(self).items()])})'


# Modifying function to handle custom postfix for keys and indices more dynamically
def convert_lines_to_tree(lines, format_dict=None):
    root = GlobalParser('root', 'key')
    current_node = root
    current_indent = -1

    format_dict = format_dict or dict()
    key_prefix = format_dict.get('key_prefix', '')
    key_postfix = format_dict.get('key_postfix', ':')
    index_prefix = format_dict.get('index_prefix', '')
    index_postfix = format_dict.get('index_postfix', ')')

    while lines:
        line = lines.pop(0)
        line = line.rstrip()
        indent = len(line)-len(line.lstrip())
        stripped_line = line.strip()

        # Conditions for custom prefix and postfix
        prefix_cond = lambda prefix: (not prefix or stripped_line.startswith(prefix))
        postfix_cond = lambda postfix: postfix in stripped_line

        while current_node and indent <= current_indent:
            current_node = current_node.parent
            current_indent -= 2  # Assuming 2 spaces for each indent level
        if prefix_cond(key_prefix) and postfix_cond(key_postfix):
            tokens = stripped_line.split(key_postfix)
            key = tokens.pop(0).strip()[len(key_prefix):]
            new_node = GlobalParser(key, 'key', current_node)
            current_node.add_child(new_node)
            current_node = new_node
            current_indent = indent
            if tokens[0]:
                indent += 2
                lines = [' '*indent+key_postfix.join(tokens).strip()]+lines
        elif prefix_cond(index_prefix) and postfix_cond(index_postfix):
            tokens = stripped_line.split(index_postfix)
            index = tokens.pop(0).strip()[len(index_prefix):]
            new_node = GlobalParser(index, 'index', current_node)
            current_node.add_child(new_node)
            current_node = new_node
            current_indent = indent
            if tokens[0]:
                indent += 2
                lines = [' '*indent+index_postfix.join(tokens).strip()]+lines
        else:
            value = convert_string_to_value(stripped_line)
            new_node = GlobalParser(value, 'value', current_node)
            current_node.add_child(new_node)
            current_node = new_node.parent
    # Remove the root node from the tree
    return root


def convert_structure_to_tree(struct, root=None, format_dict=None):
    format_dict = format_dict or dict()
    if root is None:
        root = GlobalParser('root', 'key', None, format_dict=format_dict)
    if isinstance(struct, dict):
        root.node_type = 'dict'
        value_type = 'key'
    elif isinstance(struct, Iterable) and not isinstance(struct, str):
        root.node_type = 'iter'
        value_type = 'index'
    else:
        root.node_type = 'item'
        value_node = GlobalParser(struct, 'value', root, None)
        value_node.node_type = 'value'
        root.add_child(value_node)
        return root
    for index, value in enumerate(struct):
        if root.node_type == 'dict':
            key = value
        else:
            key = index
        child_struct = struct[key]
        child = GlobalParser(key, value_type, root, format_dict)
        root.add_child(convert_structure_to_tree(child_struct, child, format_dict))
    return root


def convert_tree_to_structure(root):
    if root.node_type == 'item':
        return root.children[0].value
    elif root.node_type == 'dict':
        children_dict = {}
        for child in root.children:
            children_dict[child.value] = convert_tree_to_structure(child)
        return children_dict
    elif root.node_type == 'iter':
        size = max([child.value+1 for child in root.children])
        children_list = [None]*size
        for child in root.children:
            children_list[child.value] = convert_tree_to_structure(child)
        return children_list


def parse_lines(lines):
    decoding_formats = []
    data_lines = []
    for line in lines:
        is_decoding_format = any(
            map(lambda text: line.startswith(text), ('key-prefix', 'key-postfix', 'index-prefix', 'index-postfix'))
        )
        if is_decoding_format:
            decoding_formats.append(line)
        else:
            data_lines.append(line)
    return decoding_formats, data_lines


def parse_format(decoding_formats):
    key_prefix = index_prefix = ''
    key_postfix = ':'
    index_postfix = ')'
    for line in decoding_formats:
        line = line.strip()
        if line.startswith('key-prefix'):
            key_prefix = line[len('key-prefix -> '):].strip()
        elif line.startswith('key-postfix'):
            key_postfix = line[len('key-postfix -> '):].strip()
        elif line.startswith('index-prefix'):
            index_prefix = line[len('index-prefix -> '):].strip()
        elif line.startswith('index-postfix'):
            index_postfix = line[len('index-postfix -> '):].strip()
    return dict(key_prefix=key_prefix, key_postfix=key_postfix, index_prefix=index_prefix, index_postfix=index_postfix)


def remove_format_str(raw_str):
    return '\n'.join(parse_lines(raw_str.strip().split('\n'))[1])


def loads(raw_str):
    lines = raw_str.strip().split('\n')
    decoding_formats, lines = parse_lines(lines)
    format_dict = parse_format(decoding_formats)
    root = convert_lines_to_tree(lines, format_dict)
    return convert_tree_to_structure(root)


def dumps(obj, format_dict=None):
    return convert_structure_to_tree(obj, format_dict=format_dict).dumps()


def dump(obj, path_or_file, format_dict=None):
    if isinstance(path_or_file, (str, os.PathLike)):
        with open(path_or_file, 'w') as f:
            f.write(dumps(obj, format_dict=format_dict))
    else:
        path_or_file.write(dumps(obj, format_dict=format_dict))


def load(path_or_file):
    if isinstance(path_or_file, (str, os.PathLike)):
        with open(path_or_file, 'r') as f:
            lines = list(f.readlines())
    else:
        lines = list(path_or_file.readlines())
    decoding_formats, lines = parse_lines(lines)
    format_dict = parse_format(decoding_formats)
    root = convert_lines_to_tree(lines, format_dict=format_dict)
    return convert_tree_to_structure(root)
