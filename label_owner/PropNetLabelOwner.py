import transmission.pickle.label_owner_pb2 as label_owner_pb2
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import torch.optim as optim
from torch.nn.functional import cross_entropy
import pickle
from .utils import *
import torch
from model.utils import get_device


class PropNetLabelOwner(label_owner_pb2_grpc.LabelOwnerServiceServicer):

    def __init__(self, model, label_set, batch_size):
        self.top_model = model
        self.optimizer = optim.Adam(self.top_model.parameters(), lr=1e-3)
        self.criterion = cross_entropy
        self.label_set = label_set
        self.batch_size = batch_size
        self.num_iter = int(self.label_set.__len__() / self.batch_size)
        self.device = get_device()

        if self.label_set.__len__() % self.batch_size != 0:
            self.num_iter += 1

        self.total_loss = 0
        self.total_correct = 0

    def __get_labels(self, rnd):
        start = (rnd - 1) * self.batch_size
        end = rnd * self.batch_size
        if end > self.label_set.__len__():
            end = self.label_set.__len__()
        order_list = [i for i in range(start, end)]
        labels = self.label_set.__getitem__(order_list)[0]

        return labels

    def __fp_bp(self, middle_output, rnd):

        # middle_output = middle_output.cpu()
        middle_output.retain_grad()
        self.top_model.train()
        pred = self.top_model(middle_output)

        labels = self.__get_labels(rnd).to(self.device)
        loss = self.criterion(pred, labels.squeeze(dim=1).long())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_loss += loss.item()
        self.total_correct += get_num_correct(pred, labels.squeeze(dim=1))
        # print(pred)
        # print(labels)
        # print(self.total_correct)
        if rnd == 19:
            print("loss", self.total_loss)
            print("acc", self.total_correct / 597)
            self.total_loss = 0
            self.total_correct = 0

        middle_grad = middle_output.grad

        return middle_grad

    def top_fp_bp(self, request, context):
        middle_output = pickle.loads(request.params_msg)
        rnd = request.round
        epoch = request.epoch
        middle_grad = self.__fp_bp(middle_output, rnd)

        response = label_owner_pb2.middle_grad(round=rnd,
                                               grad_msg=pickle.dumps(middle_grad))

        if epoch % 50 == 0 and rnd == 38:
            torch.save(self.top_model.state_dict(), "../save/psn/top_model_epoch{0}.pth".format(epoch))

        return response
