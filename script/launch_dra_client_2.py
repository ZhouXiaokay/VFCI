
from trainer.dragonnet_trainer import DragonNetTrainer
from data.data_partition import DataPartition
from torch.utils.data import TensorDataset
from conf import args_parser
from model.utils import seed_torch

seed_torch()


def run(arg):
    rank = arg.rank
    csv_path = "../data/client_num2/client{0}.csv".format(rank + 1)
    dL = DataPartition(csv_path, 0.8, 4)
    datadict = dL.getDragonNetTensor(csv_path, rank)
    idx_train = datadict["DragonNet_trainData"][0]
    dragonNet_train_x = datadict["DragonNet_trainData"][1]
    dataset = TensorDataset(idx_train, dragonNet_train_x)
    n_f = dragonNet_train_x.shape[1]

    dragon_trainer = DragonNetTrainer(n_f, arg, rank, dataset)
    dragon_trainer.launch()


if __name__ == '__main__':
    args = args_parser()
    args.rank = 1
    run(args)
