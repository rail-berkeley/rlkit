"""
General purpose Python functions.

TODO(vitchyr): probably move this to its own module, not under railrl
"""
import math
import sys
import collections
import itertools


def identity(x):
    return x


def clip_magnitude(value, magnitude):
    """
    Clip the magnitude of value to be within some value.

    :param value:
    :param magnitude:
    :return:
    """
    return min(max(value, -magnitude), magnitude)


def is_numeric(x):
    return not isinstance(x, bool) and (
        isinstance(x, int) or isinstance(x, float)
    )


"""
Dictionary methods
"""


def dot_map_dict_to_nested_dict(dot_map_dict):
    """
    Convert something like
    ```
    {
        'one.two.three.four': 4,
        'one.six.seven.eight': None,
        'five.nine.ten': 10,
        'five.zero': 'foo',
    }
    ```
    into its corresponding nested dict.

    http://stackoverflow.com/questions/16547643/convert-a-list-of-delimited-strings-to-a-tree-nested-dict-using-python
    :param dot_map_dict:
    :return:
    """
    tree = {}

    for key, item in dot_map_dict.items():
        split_keys = key.split('.')
        if len(split_keys) == 1:
            if key in tree:
                raise ValueError("Duplicate key: {}".format(key))
            tree[key] = item
        else:
            t = tree
            for sub_key in split_keys[:-1]:
                t = t.setdefault(sub_key, {})
            last_key = split_keys[-1]
            if not isinstance(t, dict):
                raise TypeError(
                    "Key inside dot map must point to dictionary: {}".format(
                        key
                    )
                )
            if last_key in t:
                raise ValueError("Duplicate key: {}".format(last_key))
            t[last_key] = item
    return tree


def nested_dict_to_dot_map_dict(d, parent_key=''):
    """
    Convert a recursive dictionary into a flat, dot-map dictionary.

    :param d: e.g. {'a': {'b': 2, 'c': 3}}
    :param parent_key: Used for recursion
    :return: e.g. {'a.b': 2, 'a.c': 3}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + "." + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(nested_dict_to_dot_map_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def merge_recursive_dicts(a, b, path=None,
                          ignore_duplicate_keys_in_second_dict=False, print_info=True):
    """
    Merge two dicts that may have nested dicts.
    """
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_recursive_dicts(a[key], b[key], path + [str(key)],
                                      ignore_duplicate_keys_in_second_dict=ignore_duplicate_keys_in_second_dict,
                                      print_info=print_info)
            elif a[key] == b[key]:
                if print_info:
                    print("Same value for key: {}".format(key))
            else:
                duplicate_key = '.'.join(path + [str(key)])
                if ignore_duplicate_keys_in_second_dict:
                    if print_info:
                        print("duplicate key ignored: {}".format(duplicate_key))
                else:
                    raise Exception(
                        'Duplicate keys at {}'.format(duplicate_key)
                    )
        else:
            a[key] = b[key]
    return a


def dict_of_list__to__list_of_dicts(dict, n_items):
    """
    ```
    x = {'foo': [3, 4, 5], 'bar': [1, 2, 3]}
    ppp.dict_of_list__to__list_of_dicts(x, 3)
    # Output:
    # [
    #     {'foo': 3, 'bar': 1},
    #     {'foo': 4, 'bar': 2},
    #     {'foo': 5, 'bar': 3},
    # ]
    ```
    :param dict:
    :param n_items:
    :return:
    """
    new_dicts = [{} for _ in range(n_items)]
    for key, values in dict.items():
        for i in range(n_items):
            new_dicts[i][key] = values[i]
    return new_dicts


def list_of_dicts__to__dict_of_lists(lst, enforce_consistent_keys=True):
    """
    ```
    x = [
        {'foo': 3, 'bar': 1},
        {'foo': 4, 'bar': 2},
        {'foo': 5, 'bar': 3},
    ]
    ppp.list_of_dicts__to__dict_of_lists(x)
    # Output:
    # {'foo': [3, 4, 5], 'bar': [1, 2, 3]}
    ```
    """
    if len(lst) == 0:
        return {}
    keys = lst[0].keys()
    output_dict = collections.defaultdict(list)
    for d in lst:
        if set(d.keys()) != set(keys):
            print("dropping some keys", d.keys())
        if enforce_consistent_keys:
            assert set(d.keys()) == set(keys)
        for k in keys:
            output_dict[k].append(d[k])
    return output_dict


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


def dict_to_safe_json(d, sort=False):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    if isinstance(d, collections.OrderedDict):
        new_d = collections.OrderedDict()
    else:
        new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if (
                    isinstance(item, dict)
                    or isinstance(item, collections.OrderedDict)
            ):
                new_d[key] = dict_to_safe_json(item, sort=sort)
            else:
                new_d[key] = str(item)
    if sort:
        return collections.OrderedDict(sorted(new_d.items()))
    else:
        return new_d


def recursive_items(dictionary):
    """
    Get all (key, item) recursively in a potentially recursive dictionary.
    Usage:

    ```
    x = {
        'foo' : {
            'bar' : 5
        }
    }
    recursive_items(x)
    # output:
    # ('foo', {'bar' : 5})
    # ('bar', 5)
    ```
    :param dictionary:
    :return:
    """
    for key, value in dictionary.items():
        yield key, value
        if type(value) is dict:
            yield from recursive_items(value)


def recursive_string_replace(x, original, new):
    if isinstance(x, dict):
        return {k: recursive_string_replace(v, original, new)
                for k, v in x.items()}
    elif isinstance(x, str):
        return x.replace(original, new)
    elif isinstance(x, list):
        return [recursive_string_replace(i, original, new) for i in x]
    elif isinstance(x, tuple):
        return tuple(recursive_string_replace(i, original, new) for i in x)
    else:
        return x


def recursive_to_dict(maybe_dictable):
    """Try to convert something into a recursive dictionary.

    The main use case is to have some generic config/loading module (e.g.
    pyhocon) but to convert everything to a dictionary right before you use it.

    :param maybe_dictable: Something that may be a dictionary.
    :return:
    """
    try:
        dict_version = dict(maybe_dictable)
        return {
            k: recursive_to_dict(v)
            for k, v in dict_version.items()
        }
    except (TypeError, ValueError):
        return maybe_dictable

# TODO(vitchyr): test methods/classes below


"""
Itertools++
"""


def treemap(f, *args, atomic_type=None, **kwargs):
    """
    Recursively apply a function to a data structure.

    Usage:
    ```
    def add(x, y, constant=0):
        return x + y + constant

    tree1 = (
        [1, 2],
        {'foo': 3, 'bar': 4},
        5
    )
    tree2 = (
        [6, 7],
        {'foo': 8, 'bar': 9},
        10
    )

    treemap(add, tree1, tree2, constant=100)
    ```
    will return
    ```
    (
        [107, 109],
        {'foo': 111, 'bar': 113},
        115
    )
    ```

    Currently only supports Mapping and Iterable data structures.
    :param f: Function to apply.
    :param args: Data structures over which to apply the function.
    :param kwargs: key-word arguments that are passed to the base function
        directly.
    :return:
    """
    if len(args) == 0:
        return f(**kwargs)
    if atomic_type and isinstance(args[0], atomic_type):
        return f(*args, **kwargs)
    if isinstance(args[0], collections.Mapping):
        keys0 = args[0].keys()
        assert all(keys0 == arg.keys() for arg in args) # same keys in all args
        return type(args[0])({
            k: treemap(f, *tuple(d[k] for d in args),
                       atomic_type=atomic_type, **kwargs)
            for k in args[0]
        })
    elif isinstance(args[0], collections.Iterable):
        return type(args[0])(treemap(f, *a, atomic_type=atomic_type, **kwargs)
                             for a in zip(*args))
    else:
        return f(*args, **kwargs)


def filter_recursive(x_or_iterable):
    """
    Filter out elements that are Falsy (where bool(x) is False) from
    potentially recursive lists.

    :param x_or_iterable: An element or a list.
    :return: If x_or_iterable is not an Iterable, then return x_or_iterable.
    Otherwise, return a filtered version of x_or_iterable.
    """
    if isinstance(x_or_iterable, list):
        new_items = []
        for sub_elem in x_or_iterable:
            filtered_sub_elem = filter_recursive(sub_elem)
            if filtered_sub_elem is not None and not (
                        isinstance(filtered_sub_elem, list) and
                            len(filtered_sub_elem) == 0
            ):
                new_items.append(filtered_sub_elem)
        return new_items
    else:
        return x_or_iterable


def batch(iterable, n=1):
    """
    Split an interable into batches of size `n`. If `n` does not evenly divide
    `iterable`, the last slice will be smaller.

    https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks

    Usage:
    ```
        for i in batch(range(0,10), 3):
            print i

        [0,1,2]
        [3,4,5]
        [6,7,8]
        [9]
    ```
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def takespread(sequence, num):
    """
    Get `num` elements from the sequence that are as spread out as possible.

    https://stackoverflow.com/questions/9873626/choose-m-evenly-spaced-elements-from-a-sequence-of-length-n
    :param sequence:
    :param num:
    :return:
    """
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(math.ceil(i * length / num))]


"""
Custom Classes
"""


class IntIdDict(collections.defaultdict):
    """
    Automatically assign int IDs to hashable objects.

    Usage:
    ```
    id_map = IntIdDict()
    print(id_map['a'])
    print(id_map['b'])
    print(id_map['c'])
    print(id_map['a'])
    print(id_map['b'])
    print(id_map['a'])

    print('')

    print(id_map.get_inverse(0))
    print(id_map.get_inverse(1))
    print(id_map.get_inverse(2))
    ```

    Output:
    ```
    1
    2
    3
    1
    2
    1

    'a'
    'b'
    'c'
    ```
    :return:
    """

    def __init__(self, **kwargs):
        c = itertools.count()
        self.inverse_dict = {}
        super().__init__(lambda: next(c), **kwargs)

    def __getitem__(self, y):
        int_id = super().__getitem__(y)
        self.inverse_dict[int_id] = y
        return int_id

    def reverse_id(self, int_id):
        return self.inverse_dict[int_id]


class ConditionTimer(object):
    """
    A timer that goes off after the a fixed time period.
    The catch: you need to poll it and provide it the time!

    Usage:
    ```
    timer = PollTimer(100)  # next check will be true at 100
    timer.check(90)  # False
    timer.check(110) # True. Next check will go off at 110 + 100 = 210
    timer.check(205) # False
    timer.check(210) # True
    ```
    """

    def __init__(self, trigger_period):
        """
        :param trigger_period: If None or 0, `check` will always return False.
        """
        self.last_time_triggered = 0
        if trigger_period is None:
            trigger_period = 0
        self.trigger_period = trigger_period

    def check(self, time):
        if self.always_false:
            return False

        if time - self.last_time_triggered >= self.trigger_period:
            self.last_time_triggered = time
            return True
        else:
            return False

    @property
    def always_false(self):
        return self.trigger_period == 0


def string_tuple_to_string(strings):
    if len(strings) == 0:
        string = ""
    elif len(strings) == 1:
        string = strings[0]
    else:
        string = " ".join([str(s) for s in strings])
    return string


class _Logger(object):
    def __init__(self):
        self.n_chars = 0
        self.lines = []

    def print_over(self, *strings):
        """
        Remove anything printed in the last printover call. Then print `string`
        """
        string = string_tuple_to_string(strings)
        sys.stdout.write("\r" * self.n_chars)
        sys.stdout.write(string)
        sys.stdout.flush()
        self.n_chars = len(string)

    def newline(self):
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.n_chars = 0


line_logger = _Logger()


def find_key_recursive(obj, key):
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            result = find_key_recursive(v, key)
            if result is not None:
                return result
