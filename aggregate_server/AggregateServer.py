import torch
import grpc
import transmission.pickle.aggregate_server_pb2 as aggregate_server_pb2
import transmission.pickle.aggregate_server_pb2_grpc as aggregate_server_pb2_grpc
import transmission.pickle.label_owner_pb2 as label_owner_pb2
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import time
import torch.optim as optim
import pickle
from .utils import *
from model.utils import get_device

class AggregateServer(aggregate_server_pb2_grpc.AggregateServerServiceServicer):

    def __init__(self, num_clients, model, address):
        # initial params
        self.num_clients = num_clients
        self.sleep_time = 0.01
        self.middle_model = model
        self.device = get_device()
        self.middle_model.to(self.device)

        # for PSN
        self.optimizer = optim.Adam(self.middle_model.parameters(), lr=1e-3)

        # for DragonNet
        # self.optimizer_sgd = optim.SGD(self.middle_model.parameters(), lr=1e-5, momentum=0.9, nesterov=True,
        #                                weight_decay=0.01)
        # self.optimizer = optim.Adam(self.middle_model.parameters(), lr=1e-3, weight_decay=0.01)

        self.max_msg_size = 1000000000
        self.options = [('grpc.max_send_message_length', self.max_msg_size),
                        ('grpc.max_receive_message_length', self.max_msg_size)]
        channel = grpc.insecure_channel(address, options=self.options)
        self.stub = label_owner_pb2_grpc.LabelOwnerServiceStub(channel)

        # for sum_encrypted
        self.n_sum_request = 0
        self.n_sum_response = 0
        self.n_sum_round = 0
        self.bottom_output_dict = {}
        self.bottom_output_list = []
        self.bottom_grad_list = []
        self.bp_completed = False

    def __reset_sum(self):
        self.bp_completed = False
        self.bottom_output_dict.clear()
        self.bottom_output_list.clear()
        self.bottom_grad_list.clear()
        self.n_sum_request = 0
        self.n_sum_response = 0

    def __concat_bottom(self):
        for i in range(self.num_clients):
            self.bottom_output_list.append(self.bottom_output_dict[i])

        concat_bottom = torch.cat(self.bottom_output_list, dim=-1)

        return concat_bottom

    def __split_bottom_grads(self, bottom_grads):
        shape_list = []
        for tensor in self.bottom_output_list:
            if len(tensor.shape) == 1:
                shape_list.append(tensor.shape[0])
            else:
                shape_list.append(tensor.shape[1])

        return list(torch.split(bottom_grads, shape_list, dim=-1))

    def middle_fp_bp(self, request, context):
        bottom_vector = pickle.loads(request.params_msg)
        client_rank = request.client_rank
        rnd = request.round
        epoch = request.epoch
        self.bottom_output_dict[client_rank] = bottom_vector
        self.n_sum_request += 1

        # wait until receiving of all clients' requests
        wait_start = time.time()
        while self.n_sum_request % self.num_clients != 0:
            time.sleep(self.sleep_time)
        wait_time = time.time() - wait_start

        if client_rank == self.num_clients - 1:

            concat_bottom = self.__concat_bottom()
            # concat_bottom = concat_bottom.cpu()
            middle_output = middle_model_fp(self.middle_model, concat_bottom)
            request = label_owner_pb2.middle_output(round=rnd,
                                                    epoch=epoch,
                                                    params_msg=pickle.dumps(middle_output))

            response = self.stub.top_fp_bp(request)
            middle_grad = pickle.loads(response.grad_msg)
            assert rnd == response.round

            all_bottom_grads = middle_model_bp(middle_grad, self.optimizer, middle_output, concat_bottom)

            self.bottom_grad_list = self.__split_bottom_grads(all_bottom_grads)

            if epoch % 50 == 0 and rnd == 110:
                torch.save(self.middle_model.state_dict(), "../save/dcn_pd/dcn_middle_model_epoch{0}.pth".format(epoch))
                if epoch == 200:
                    self.optimizer = optim.SGD(self.middle_model.parameters(), lr=1e-5, momentum=0.9, nesterov=True,
                                               weight_decay=0.01)
            self.bp_completed = True

        while not self.bp_completed:
            time.sleep(self.sleep_time)

        # wait until all response is make
        bottom_grad = self.bottom_grad_list[client_rank]
        response = aggregate_server_pb2.bottom_grad(client_rank=client_rank,
                                                    grad_msg=pickle.dumps(bottom_grad))
        self.n_sum_response = self.n_sum_response + 1
        while self.n_sum_response % self.num_clients != 0:
            time.sleep(self.sleep_time)

        if client_rank == self.num_clients - 1:
            self.__reset_sum()

        # wait until cache for sum is reset
        self.n_sum_round = self.n_sum_round + 1
        while self.n_sum_round % self.num_clients != 0:
            time.sleep(self.sleep_time)

        return response
