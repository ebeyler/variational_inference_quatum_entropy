def unique(list):
    """Return list of unique elements.

    Args:
        list (list): list to process

    Returns:
        list: list of unique elements
    """
    output = []
    for x in list:
        if x not in output:
            output.append(x)
    return output

def unique(list):
    """Remove redundant elements from a list.

    Args:
        list (list)

    Returns:
        list
    """
    output = []
    for x in list:
        if x not in output:
            output.append(x)
    return output

def diff_list(list1,list2):
    """Return list of elements in list1 that are not in list2.

    Args:
        list1 (list): First list
        list2 (list): Second list

    Returns:
        list
    """
    output = []
    for x in list1:
        if x not in list2:
            output.append(x)
    return output