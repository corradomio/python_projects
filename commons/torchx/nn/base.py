import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
# Used to insert breakpoints inside the layers

class Probe(nn.Module):

    def __init__(self, name=""):
        super().__init__()
        self.name = name
        self._log = True
        self._repr = f"Probe({name})"

    def forward(self, input):
        if self._log:
            print(f"{self._repr}: {tuple(input.shape)}")
            self._log = False
        return input

    def __repr__(self):
        return self._repr
# end


# ---------------------------------------------------------------------------
# Select
# ---------------------------------------------------------------------------
# If the input is a tuple/list o an hierarchical structure, it permits to
# select an element

class Select(nn.Module):

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
# MultiInputs
# ---------------------------------------------------------------------------
# 1) apply the 'input_models' to the list of input tensors
# 2) concatenate all generated output tensors in a single tensor
# 3) pass the generated tensor to the 'output model'
# 4) return the output of the last model

class MultiInputs(nn.Module):

    def __init__(self, input_models, output_model):
        super().__init__()
        self.input_models = input_models
        self.output_model = output_model
        self.n_inputs = len(input_models)

    def forward(self, input_list):
        assert len(input_list) == self.n_inputs
        n = self.n_inputs
        inner_results = []

        # for each input, call the related model
        for i in range(n):
            input = input_list[i]
            model = self.input_models[i]

            inner_result = model.forward(input)
            inner_results.append(inner_result)

        # concatenate the inner results
        inner = torch.concatenate(inner_results, dim=1)

        result = self.output_model.forward(inner)
        return result
    # end
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------