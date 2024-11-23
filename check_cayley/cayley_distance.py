from itertools import permutations

def cayley_distance(perm1, perm2):
    """
    Computes the Cayley distance between two permutations.

    Args:
        perm1 (list): The first permutation as a list of integers.
        perm2 (list): The second permutation as a list of integers.

    Returns:
        int: The Cayley distance (number of transpositions required).
    """
    if sorted(perm1) != sorted(perm2):
        raise ValueError("Input permutations must contain the same elements.")

    # Convert to cycle representation
    visited = [False] * len(perm1)
    cycles = 0

    perm1_index = {v: i for i, v in enumerate(perm1)}
    current_perm = [perm1_index[elem] for elem in perm2]

    for i in range(len(current_perm)):
        if not visited[i]:
            cycles += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = current_perm[j]

    return len(perm1) - cycles

# Example Usage
perm1 = [1, 2, 3, 4]
perm2 = [4, 3, 2, 1]
distance = cayley_distance(perm1, perm2)
print(f"Cayley distance between {perm1} and {perm2}: {distance}")
