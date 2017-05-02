# Set of generally useful functions.


def chunk_list(my_list, n):
    # generator to chunk list into lists of max length of n
    for i in range(0, len(my_list), n):
        yield my_list[i:i + n]


def flatten(list_of_lists):
    # flatten a list of lists into a single list
    return [item for sublist in list_of_lists for item in sublist]


def jaccard_distance(set1, set2):
    # if both sets are empty
    if len(set1) + len(set2) == 0:
        return 0.0
    # otherwise (at least one set non-empty)
    else:
        return float(len(set1 & set2))/float(len(set1 | set2))
