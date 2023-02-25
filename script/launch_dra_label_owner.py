
import torch
from torch.utils.data import TensorDataset
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import grpc
from concurrent import futures
from model.DragonNet import TopModel
from conf import args_parser
from label_owner.DraNetLabelOwner import DraNetLabelOwner
from data.data_partition import DataPartition

from model.utils import seed_torch,get_device

seed_torch()


def launch_label_owner_server(label_set):
    device = get_device()
    args = args_parser()
    address = args.label_owner_address
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    model = TopModel().to(device)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=options)
    label_owner_pb2_grpc.add_LabelOwnerServiceServicer_to_server(
        DraNetLabelOwner(model, label_set, args.batch_size),
        server)
    server.add_insecure_port(address)
    server.start()
    print("DragonNet Label Owner start")
    server.wait_for_termination()

if __name__ == '__main__':
    csv_path = "../data/client_num2/client2.csv"
    dL = DataPartition(csv_path, 0.8)
    datadict = dL.getDragonNetTensor(csv_path, 1)
    train_y_ycf_t = datadict["DragonNet_trainData"][2]
    train_y_t = torch.stack((train_y_ycf_t[:, 0], train_y_ycf_t[:, 2]), dim=1).cuda()
    label_set = torch.utils.data.TensorDataset(train_y_t)
    # print(label_set.__getitem__([0, 1]))
    launch_label_owner_server(label_set)
