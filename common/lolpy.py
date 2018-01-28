def remove_if_exists(some_list: list, remove_values):
    for remove_value in remove_values:
        if remove_value in some_list:
            some_list.remove(remove_value)


def get_int_percentage(colA, colB):
    return (colA / colB).multiply(100).round()