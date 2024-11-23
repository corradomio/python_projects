from bisect import bisect_left

def ulam_distance(perm1, perm2):
    """
    Computes the Ulam distance between two permutations.

    Args:
        perm1 (list): The first permutation as a list of integers.
        perm2 (list): The second permutation as a list of integers.

    Returns:
        int: The Ulam distance (minimum number of deletions required).
    """
    if sorted(perm1) != sorted(perm2):
        raise ValueError("Input permutations must contain the same elements.")

    # Map perm1 to perm2's indices to create an isomorphic subsequence problem
    index_map = {value: i for i, value in enumerate(perm2)}
    mapped_perm = [index_map[value] for value in perm1]

    # Find the length of the longest increasing subsequence (LIS)
    lis = []
    for value in mapped_perm:
        pos = bisect_left(lis, value)
        if pos == len(lis):
            lis.append(value)
        else:
            lis[pos] = value

    # Ulam distance = length of permutation - length of LIS
    return len(perm1) - len(lis)

# Example Usage
perm1 = [1, 3, 2, 4]
perm2 = [3, 1, 4, 2]
distance = ulam_distance(perm1, perm2)
print(f"Ulam distance between {perm1} and {perm2}: {distance}")
