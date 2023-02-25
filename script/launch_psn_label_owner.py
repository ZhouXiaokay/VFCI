import torch
from torch.utils.data import TensorDataset
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import grpc
from concurrent import futures
from model.PropensityNet import TopModel
from conf import args_parser
from label_owner.PropNetLabelOwner import PropNetLabelOwner
from data.data_partition import DataPartition
from model.utils import get_device


def launch_label_owner_server(label_set):
    device = get_device()
    args = args_parser()
    address = args.label_owner_address
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    model = TopModel("train").to(device)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=options)
    label_owner_pb2_grpc.add_LabelOwnerServiceServicer_to_server(
        PropNetLabelOwner(model, label_set, args.batch_size),
        server)
    server.add_insecure_port(address)
    server.start()
    print("PSN Label Owner start")
    server.wait_for_termination()


if __name__ == '__main__':
    csv_path = "../data/client_num2/client2.csv"
    dL = DataPartition(csv_path, 0.8, 4)
    datadict = dL.getPSNTensor(csv_path, True)

    train_t = datadict["PSN_trainData"][2]
    label_set = torch.utils.data.TensorDataset(train_t)
    # print(label_set.__getitem__([0, 1, 2, 3])[0])

    launch_label_owner_server(label_set)
