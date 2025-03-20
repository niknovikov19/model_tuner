from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .input_nd import PopInputND, NetInputND


class OUDict(dict):
    """ A dictionary that only allows keys 'ou_mean' and 'ou_std'. """

    __allowed_keys = {'ou_mean', 'ou_std'}

    def __setitem__(self, key, value):
        # Only allow ou_mean, ou_std
        if key not in self.__allowed_keys:
            raise KeyError(
                f"Key '{key}' not permitted. Must be one of: {self.__allowed_keys}"
            )
        super().__setitem__(key, value)

    def __delitem__(self, key):
        # Disallow removing existing keys
        raise NotImplementedError('Deletion of keys is not allowed in OUDict.')


@dataclass
class PopInputOU(PopInputND):
    """Ornstein-Uhlenbeck input with ou_mean and ou_std params. """

    def __init__(self, ou_mean: float = 0, ou_std: float = 0, **kwargs):
        # Create a restricted OUDict instance up-front.
        super().__init__(vars=OUDict(), **kwargs)
        # Explicitly set the two allowed keys.
        self.vars['ou_mean'] = ou_mean
        self.vars['ou_std']  = ou_std

    # Provide property getters/setters for convenience:
    @property
    def ou_mean(self) -> float:
        return self.vars['ou_mean']

    @ou_mean.setter
    def ou_mean(self, val: float):
        self.vars['ou_mean'] = val

    @property
    def ou_std(self) -> float:
        return self.vars['ou_std']

    @ou_std.setter
    def ou_std(self, val: float):
        self.vars['ou_std'] = val


@dataclass
class NetInputOU(NetInputND):
    """NetInput for Ornstein-Uhlenbeck inputs. """

    def get_pop_ou_mean_vec(self) -> np.ndarray:
        return self.get_pop_inputs_vec('ou_mean')
    
    def get_pop_ou_std_vec(self) -> np.ndarray:
        return self.get_pop_inputs_vec('ou_std')