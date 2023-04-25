import torch
import torch.utils.data
import torch.nn.functional as F
from datasets import load_dataset
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

model = torch.nn.Transformer().to(device)
optimizer = torch.optim.Adam(model.parameters())

dataset = load_dataset('my_dataset')
data = torch.utils.data.DataLoader(dataset, shuffle=True)

model, optimizer, data = accelerator.prepare(model, optimizer, data)

model.train()
for epoch in range(10):
    for source, targets in data:
        source = source.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = model(source)
        loss = F.cross_entropy(output, targets)

        # loss.backward()
        accelerator.backward(loss)

        optimizer.step()
# end
