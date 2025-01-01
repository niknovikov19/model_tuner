def from_dict_or_dataclass(x, cls_type):
    if isinstance(x, cls_type):
        return x
    elif isinstance(x, dict):
       return cls_type(**x)
    else:
        raise TypeError(f'Cannot convert {type(x)} to {type(cls_type)}')