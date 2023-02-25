from trainer.propensity_trainer import PropNetTrainer
from data.data_partition import DataPartition
from torch.utils.data import TensorDataset
from conf import args_parser


def run(arg):
    rank = arg.rank
    csv_path = "../data/client_num2/client{0}.csv".format(rank + 1)
    dL = DataPartition(csv_path, 0.8, 4)
    datadict = dL.getPSNTensor(csv_path, rank)
    idx_train = datadict["PSN_trainData"][0]
    ps_train_x = datadict["PSN_trainData"][1]
    dataset = TensorDataset(idx_train, ps_train_x)
    n_f = ps_train_x.shape[1]

    prop_trainer = PropNetTrainer(n_f, arg, rank, dataset)
    prop_trainer.launch()


if __name__ == '__main__':
    args = args_parser()
    args.rank = 1
    run(args)
