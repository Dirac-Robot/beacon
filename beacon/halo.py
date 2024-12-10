from typing import Iterable, Iterator, Generator, Sequence
from beacon.adict import ADict


class HALo:
    _curl = None

    def __init__(self, *sources, collate_fn=None, sentinel=None):
        if len(sources) == 1:
            src = sources[0]
        else:
            src = zip(*sources)
        if hasattr(src, '__len__'):
            self.last_index = len(src)-1
        else:
            self.last_index = None
        if callable(src) and sentinel:
            src = iter(src, sentinel)
        elif not isinstance(src, Iterable):
            raise TypeError(f'{self.__class__.__name__} wraps iterable objects only.')
        self.src = src
        self.collate_fn = collate_fn
        self._index = -1
        self.next = None
        self.prev_loop = None

    def __iter__(self):
        return self

    def __next__(self):
        if HALo._curl != self:
            self.prev_loop = HALo._curl
            HALo._curl = self
        self._index += 1
        if self.next is not None:
            output, self.next = self.next, None
        else:
            try:
                if isinstance(self.src, Iterator) or isinstance(self.src, Generator):
                    output = next(self.src)
                elif isinstance(self.src, Sequence):
                    output = self.src[self._index]
                else:
                    raise TypeError(f'Source of {self.__class__.__name__} seems to be not an iterable object.')
            except StopIteration:
                HALo._curl = self.prev_loop
                raise StopIteration
        if self.collate_fn is not None:
            output = self.collate_fn(output)
        return output

    @classmethod
    def curl(cls):
        return cls._curl

    def __contains__(self, item):
        return self.curl is self

    @property
    def first(self):
        return self._index == 0

    @property
    def last(self):
        if self.last_index is not None:
            is_last = self._index == self.last_index
        elif self.next is not None:
            is_last = True
        else:
            try:
                self.next = next(self)
                is_last = True
            except StopIteration:
                is_last = False
        return is_last

    @property
    def index(self):
        return self._index

    def reset(self):
        self._index = -1
        self.next = None
        self.src = iter(self.src)

    def where(self, items=None):
        indices = ADict(default=[])
        for elem in self:
            for item in items:
                if elem == item:
                    indices[elem].append(self.index)
        self.reset()
        return indices


halo = HALo


class HALoFunctional:
    @classmethod
    def curl(cls):
        return halo.curl()

    @classmethod
    def current_loop(cls):
        return halo.curl()

    @classmethod
    def ci(cls):
        return halo.curl().index

    @classmethod
    def first(cls):
        return halo.curl().first

    @classmethod
    def last(cls):
        return halo.curl().last


functional = HALoFunctional
