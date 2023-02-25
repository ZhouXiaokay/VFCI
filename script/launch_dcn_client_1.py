from trainer.dcn_trainer import DCNTrainer
from data.data_partition import DataPartition
from conf import args_parser

from model.utils import seed_torch

seed_torch()


def run(arg):
    rank = arg.rank
    csv_path = "../data/client_num2/client{0}.csv".format(rank + 1)
    csv_path_2 = "../data/client_num2/client2.csv"
    dL = DataPartition(csv_path, 0.8)
    index = dL.getIndex(csv_path_2)
    datadict = dL.getDCNTensorWithoutLabel(csv_path, "train", index)

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
    print(treated_idx)



if __name__ == '__main__':
    args = args_parser()
    run(args)
