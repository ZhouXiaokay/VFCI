
import torch
from torch.utils.data import TensorDataset
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import grpc
from concurrent import futures
from model.DCN import TopModel
from conf import args_parser
from label_owner.DCNLabelOwner import DCNLabelOwner
from data.data_partition import DataPartition
from model.PropensityNet import BottomModel
from model.utils import seed_torch, get_propensity_score,get_device

seed_torch()


def launch_label_owner_server(treated_set, control_set):
    device =get_device()
    args = args_parser()
    address = args.label_owner_address
    max_msg_size = 1000000000
    options = [('grpc.max_send_message_length', max_msg_size), ('grpc.max_receive_message_length', max_msg_size)]
    model = TopModel(True).to(device)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5), options=options)
    label_owner_pb2_grpc.add_LabelOwnerServiceServicer_to_server(
        DCNLabelOwner(model, treated_set, control_set),
        server)
    server.add_insecure_port(address)
    server.start()
    print("DCN Label Owner start")
    server.wait_for_termination()


def get_bottom_concat(x_1, x_2):
    device = get_device()
    x_1 = x_1.to(device)
    x_2 = x_2.to(device)
    bottom_model_1 = BottomModel(13).to(device)
    bottom_model_2 = BottomModel(12).to(device)
    bottom_model_1.load_state_dict(torch.load("../save/psn/bottom_model_epoch50_client_0.pth"))
    bottom_model_2.load_state_dict(torch.load("../save/psn/bottom_model_epoch50_client_1.pth"))
    bottom_model_1.eval()
    bottom_model_2.eval()
    bottom_out_1 = bottom_model_1(x_1)
    bottom_out_2 = bottom_model_2(x_2)
    concat_bottom = torch.cat([bottom_out_1, bottom_out_2], dim=1)
    return concat_bottom


if __name__ == '__main__':
    csv_path_1 = "../data/client_num2/client1.csv"
    csv_path_2 = "../data/client_num2/client2.csv"
    dL = DataPartition(csv_path_2, 0.8)
    index = dL.getIndex(csv_path_2)
    datadict_1 = dL.getDCNTensorWithoutLabel(csv_path_1, "train", index)

    datadict_2 = dL.getDCNTensorWithLabel(csv_path_2, "train")
    treated_data_2 = datadict_2["treat_trainData"]
    control_data_2 = datadict_2["control_trainData"]

    treated_data_1 = datadict_1["treat_trainData"]
    control_data_1 = datadict_1["control_trainData"]

    treated_id = treated_data_2[0]
    control_id = control_data_2[0]

    treated_x_1 = treated_data_1[1]
    control_x_1 = control_data_1[1]
    treated_x_2 = treated_data_2[1]
    control_x_2 = control_data_2[1]

    concat_bottom_treated = get_bottom_concat(treated_x_1, treated_x_2)
    concat_bottom_control = get_bottom_concat(control_x_1, control_x_2)

    treated_ps = get_propensity_score("../save/psn/", concat_bottom_treated)
    control_ps = get_propensity_score("../save/psn/", concat_bottom_control)

    treated_outcome_y = treated_data_2[3]
    control_outcome_y = control_data_2[3]
    # treated_set = torch.utils.data.TensorDataset(treated_outcome_y, treated_ps)
    # control_set = torch.utils.data.TensorDataset(control_outcome_y, control_ps)

    treated_y = torch.cat([treated_outcome_y, treated_id], dim=1)
    control_y = torch.cat([control_outcome_y, control_id], dim=1)
    treated_set = torch.utils.data.TensorDataset(treated_y, treated_ps)
    control_set = torch.utils.data.TensorDataset(control_y, control_ps)

    launch_label_owner_server(treated_set, control_set)
