from itertools import permutations

def is_valid_permutation(a, b):
    """Checks if a and b are valid permutations of the same elements."""
    return sorted(a) == sorted(b)

def transposition_distance(a, b):
    """
    Computes the transposition distance between two permutations.
    Returns the minimum number of adjacent swaps needed to transform a into b.
    """
    if not is_valid_permutation(a, b):
        raise ValueError("Input sequences must be permutations of the same elements.")

    n = len(a)
    distance = 0
    a = list(a)

    for i in range(n):
        # If the current element is already in the correct position, skip it
        if a[i] != b[i]:
            # Find the index in `a` that matches `b[i]`
            target_index = a.index(b[i])
            # Perform a transposition to bring the correct element closer
            a[i:target_index + 1] = [a[target_index]] + a[i:target_index]
            distance += 1

    return distance

# Example usage
a = [1, 3, 4, 2]
b = [4, 3, 2, 1]
print(f"Transposition distance between {a} and {b}: {transposition_distance(a, b)}")
