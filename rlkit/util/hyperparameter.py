"""
Custom hyperparameter functions.
"""
import abc
import copy
import math
import random
import itertools
from typing import List

import rlkit.pythonplusplus as ppp


class Hyperparameter(metaclass=abc.ABCMeta):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


class RandomHyperparameter(Hyperparameter):
    def __init__(self, name):
        super().__init__(name)
        self._last_value = None

    @abc.abstractmethod
    def generate_next_value(self):
        """Return a value for the hyperparameter"""
        return

    def generate(self):
        self._last_value = self.generate_next_value()
        return self._last_value


class EnumParam(RandomHyperparameter):
    def __init__(self, name, possible_values):
        super().__init__(name)
        self.possible_values = possible_values

    def generate_next_value(self):
        return random.choice(self.possible_values)


class LogFloatParam(RandomHyperparameter):
    """
    Return something ranging from [min_value + offset, max_value + offset],
    distributed with a log.
    """
    def __init__(self, name, min_value, max_value, *, offset=0):
        super(LogFloatParam, self).__init__(name)
        self._linear_float_param = LinearFloatParam("log_" + name,
                                                    math.log(min_value),
                                                    math.log(max_value))
        self.offset = offset

    def generate_next_value(self):
        return math.e ** (self._linear_float_param.generate()) + self.offset


class LinearFloatParam(RandomHyperparameter):
    def __init__(self, name, min_value, max_value):
        super(LinearFloatParam, self).__init__(name)
        self._min = min_value
        self._delta = max_value - min_value

    def generate_next_value(self):
        return random.random() * self._delta + self._min


class LogIntParam(RandomHyperparameter):
    def __init__(self, name, min_value, max_value, *, offset=0):
        super().__init__(name)
        self._linear_float_param = LinearFloatParam("log_" + name,
                                                    math.log(min_value),
                                                    math.log(max_value))
        self.offset = offset

    def generate_next_value(self):
        return int(
            math.e ** (self._linear_float_param.generate()) + self.offset
        )


class LinearIntParam(RandomHyperparameter):
    def __init__(self, name, min_value, max_value):
        super(LinearIntParam, self).__init__(name)
        self._min = min_value
        self._max = max_value

    def generate_next_value(self):
        return random.randint(self._min, self._max)


class FixedParam(RandomHyperparameter):
    def __init__(self, name, value):
        super().__init__(name)
        self._value = value

    def generate_next_value(self):
        return self._value


class Sweeper(object):
    pass


class RandomHyperparameterSweeper(Sweeper):
    def __init__(self, hyperparameters=None, default_kwargs=None):
        if default_kwargs is None:
            default_kwargs = {}
        self._hyperparameters = hyperparameters or []
        self._validate_hyperparameters()
        self._default_kwargs = default_kwargs

    def _validate_hyperparameters(self):
        names = set()
        for hp in self._hyperparameters:
            name = hp.name
            if name in names:
                raise Exception("Hyperparameter '{0}' already added.".format(
                    name))
            names.add(name)

    def set_default_parameters(self, default_kwargs):
        self._default_kwargs = default_kwargs

    def generate_random_hyperparameters(self):
        hyperparameters = {}
        for hp in self._hyperparameters:
            hyperparameters[hp.name] = hp.generate()
        hyperparameters = ppp.dot_map_dict_to_nested_dict(hyperparameters)
        return ppp.merge_recursive_dicts(
            hyperparameters,
            copy.deepcopy(self._default_kwargs),
            ignore_duplicate_keys_in_second_dict=True,
        )

    def sweep_hyperparameters(self, function, num_configs):
        returned_value_and_params = []
        for _ in range(num_configs):
            kwargs = self.generate_random_hyperparameters()
            score = function(**kwargs)
            returned_value_and_params.append((score, kwargs))

        return returned_value_and_params


class DeterministicHyperparameterSweeper(Sweeper):
    """
    Do a grid search over hyperparameters based on a predefined set of
    hyperparameters.
    """
    def __init__(self, hyperparameters, default_parameters=None):
        """

        :param hyperparameters: A dictionary of the form
        ```
        {
            'hp_1': [value1, value2, value3],
            'hp_2': [value1, value2, value3],
            ...
        }
        ```
        This format is like the param_grid in SciKit-Learn:
        http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search
        :param default_parameters: Default key-value pairs to add to the
        dictionary.
        """
        self._hyperparameters = hyperparameters
        self._default_kwargs = default_parameters or {}
        named_hyperparameters = []
        for name, values in self._hyperparameters.items():
            named_hyperparameters.append(
                [(name, v) for v in values]
            )
        self._hyperparameters_dicts = [
            ppp.dot_map_dict_to_nested_dict(dict(tuple_list))
            for tuple_list in itertools.product(*named_hyperparameters)
        ]

    def iterate_hyperparameters(self):
        """
        Iterate over the hyperparameters in a grid-manner.

        :return: List of dictionaries. Each dictionary is a map from name to
        hyperpameter.
        """
        return [
            ppp.merge_recursive_dicts(
                hyperparameters,
                copy.deepcopy(self._default_kwargs),
                ignore_duplicate_keys_in_second_dict=True,
            )
            for hyperparameters in self._hyperparameters_dicts
        ]


# TODO(vpong): Test this
class DeterministicSweeperCombiner(object):
    """
    A simple wrapper to combiner multiple DeterministicHyperParameterSweeper's
    """
    def __init__(self, sweepers: List[DeterministicHyperparameterSweeper]):
        self._sweepers = sweepers

    def iterate_list_of_hyperparameters(self):
        """
        Usage:

        ```
        sweeper1 = DeterministicHyperparameterSweeper(...)
        sweeper2 = DeterministicHyperparameterSweeper(...)
        combiner = DeterministicSweeperCombiner([sweeper1, sweeper2])

        for params_1, params_2 in combiner.iterate_list_of_hyperparameters():
            # param_1 = {...}
            # param_2 = {...}
        ```
        :return: Generator of hyperparameters, in the same order as provided
        sweepers.
        """
        return itertools.product(
            sweeper.iterate_hyperparameters()
            for sweeper in self._sweepers
        )


def recursive_dictionary_update(d, u):
    """Recursive d.update(u) for dictionaries d, u"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dictionary_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
