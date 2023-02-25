import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from model.utils import get_propensity_score
from model.PropensityNet import BottomModel, TopModel
import numpy as np

x_1 = torch.tensor([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]])
x_2 = torch.tensor([[2, 2, 2], [2, 2, 2]])
x_3 = torch.tensor([[3, 3, 3], [3, 3, 3]])

batch_size = 2
dataset = TensorDataset(x_2)
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# for batch in data_loader:
#     print(batch)
# order = int(dataset.__len__() / batch_size)
# if dataset.__len__() % batch_size != 0:
#     order += 1

print(torch.split(dataset.__getitem__([0])[0].squeeze(), [2, 1]))

model = TopModel("train").cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_x = torch.tensor([6.2964, 0.2476]).chunk(2, 0)

print(test_x)

data = np.load("data/ihdp/ihdp_npci_1-1000.train.npz")
print(data['x'][:, :, 0].shape)
