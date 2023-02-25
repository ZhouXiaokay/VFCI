from trainer.dcn_trainer import DCNTrainer
from data.data_partition import DataPartition
from conf import args_parser

from model.utils import seed_torch

seed_torch()


def run(arg):
    rank = arg.rank
    csv_path = "../data/client_num2/client{0}.csv".format(rank + 1)

    dL = DataPartition(csv_path, 0.8)
    datadict = dL.getDCNTensorWithLabel(csv_path, "train")

    treated_data = datadict["treat_trainData"]
    control_data = datadict["control_trainData"]
    treated_idx = treated_data[0]
    control_idx = control_data[0]
    treated_x = treated_data[1]
    control_x = control_data[1]

    dataset = {"treated": [treated_idx, treated_x], "control": [control_idx, control_x]}
    n_f = treated_x.shape[1]

    dcn_trainer = DCNTrainer(n_f, arg, rank, dataset)
    dcn_trainer.launch()



if __name__ == '__main__':
    args = args_parser()
    args.rank = 1
    run(args)
