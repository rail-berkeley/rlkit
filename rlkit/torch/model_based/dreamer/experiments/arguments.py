import argparse


def convert_value_to_type(value, type_):
    if type_ == "bool":
        value = value == "True"
    elif type_ == "int":
        value = int(value)
    elif type_ == "float":
        value = float(value)
    elif type_ == "str":
        value = value  # default is just string
    else:
        raise NotImplementedError("type {} not implemented".format(type_))
    return value


def get_args():
    parser = argparse.ArgumentParser(description="Experiment Launcher Arguments")

    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("-sk", "--search_keys", type=str, nargs="*", default=[])
    parser.add_argument(
        "-sv", "--search_values", nargs="*", default=[], action="append"
    )
    parser.add_argument("-st", "--search_types", nargs="*", default=[])

    # parse arguments
    args = parser.parse_args()
    if args.debug:
        args.exp_prefix = "test" + args.exp_prefix
    updated_search_values = []
    for value, type_ in zip(args.search_values, args.search_types):
        if type(value) == list:
            updated_value = []
            for nested_value in value:
                nested_value = convert_value_to_type(nested_value, type_)
                updated_value.append(nested_value)
        else:
            updated_value = convert_value_to_type(value, type_)
        updated_search_values.append(updated_value)
    args.search_values = updated_search_values

    return args
