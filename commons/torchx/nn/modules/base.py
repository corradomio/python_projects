import torch.nn as nn


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
#

def print_shape(what, x, i=0):
    if isinstance(x, (list, tuple)):
        if i == 0:
            print("  "*i, what, "...")
        else:
            print("  " * i, "...")
        for t in x:
            print_shape(what, t, i+1)
        return
    if i == 0:
        print("  "*i, what, tuple(x.shape))
    else:
        print("  " * i, tuple(x.shape))


class Probe(nn.Module):
    """
    Used to insert breakpoints during the training/prediction and to print the
    tensor shapes (ONLY the first time)
    """

    def __init__(self, name="probe"):
        super().__init__()
        self.name = name
        self._log = True
        self._repr = f"[{name}]\t"

    def forward(self, input):
        if self._log:
            print_shape(self._repr, input)
            self._log = False
        return input

    def __repr__(self):
        return self._repr
# end


# ---------------------------------------------------------------------------
# Select
# ---------------------------------------------------------------------------
#

class Select(nn.Module):
    """
    If  'input' is a tuple/list o an hierarchical structure, it permits to
    select an element based on a sequence of indices:

        select=(3,2,4,1)

    is converted into

        input[3][2][4][1]
    """

    def __init__(self, select=()):
        super().__init__()
        assert isinstance(select, (int, list, tuple))
        self.select = [select] if isinstance(select, int) else list(select)

    def forward(self, input):
        output = input
        for s in self.select:
            output = output[s]
        return output
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
