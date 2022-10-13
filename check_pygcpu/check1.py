from pprint import pprint
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

pprint(len(dataset))
pprint(dataset.num_classes)
pprint(dataset.num_node_features)

data = dataset[0]
pprint(data)

pprint(data.is_undirected())

train_dataset = dataset[:540]
pprint(train_dataset)

test_dataset = dataset[540:]
pprint(test_dataset)

dataset = dataset.shuffle()
pprint(dataset)
