from dataclasses import dataclass, field, fields
from typing import Any


class DataClassDictMixin(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class mxDataclass(DataClassDictMixin):
    def items(self):
        return [(field.name, getattr(self, field.name)) for field in fields(self)]
