def lee_distance(a, b, q):
    """
    Computes the Lee distance between two integers in modular space.

    Args:
        a (int): The first integer.
        b (int): The second integer.
        q (int): The modulus (size of the circular space).

    Returns:
        int: The Lee distance between a and b.
    """
    if not (0 <= a < q and 0 <= b < q):
        raise ValueError("Inputs a and b must be in the range [0, q-1].")

    diff = abs(a - b)
    return min(diff, q - diff)

def lee_distance_vector(v1, v2, q):
    """
    Computes the Lee distance between two vectors in modular space.

    Args:
        v1 (list): The first vector as a list of integers.
        v2 (list): The second vector as a list of integers.
        q (int): The modulus (size of the circular space).

    Returns:
        int: The total Lee distance between the two vectors.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors v1 and v2 must have the same length.")

    return sum(lee_distance(a, b, q) for a, b in zip(v1, v2))

# Example Usage
a, b, q = 2, 7, 10  # Single elements in modular space
print(f"Lee distance between {a} and {b} in Z_{q}: {lee_distance(a, b, q)}")

v1 = [1, 3, 5]
v2 = [3, 8, 0]
q = 10  # Modulus for the vector elements
print(f"Lee distance between {v1} and {v2} in Z_{q}: {lee_distance_vector(v1, v2, q)}")
