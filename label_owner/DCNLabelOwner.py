import transmission.pickle.label_owner_pb2 as label_owner_pb2
import transmission.pickle.label_owner_pb2_grpc as label_owner_pb2_grpc
import torch.optim as optim
import torch.nn as nn
import pickle
import torch
from label_owner.utils import dcn_fp
from model.utils import get_device


class DCNLabelOwner(label_owner_pb2_grpc.LabelOwnerServiceServicer):

    def __init__(self, model, treated_set, control_set):
        self.device = get_device()
        self.top_model = model.to(self.device)
        self.optimizer = optim.Adam(self.top_model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.treated_set = treated_set
        self.control_set = control_set

        self.total_loss = 0

    def __get_labels(self, rnd, epoch):
        p_score, true_ite = None, None
        if epoch % 2 == 0:
            outcome_id, p_score = self.treated_set.__getitem__(rnd)
            outcome, idx = torch.split(outcome_id, [2, 1])
            y_f, y_cf = outcome.chunk(2, 0)
            true_ite = y_f - y_cf
            # print(idx)
        elif epoch % 2 == 1:
            outcome_id, p_score = self.control_set.__getitem__(rnd)
            outcome, idx = torch.split(outcome_id, [2, 1])
            y_f, y_cf = outcome.chunk(2, 0)
            true_ite = y_cf - y_f
            # print(idx)
        return true_ite, p_score

    def __fp_bp(self, middle_output, rnd, epoch):

        middle_output.retain_grad()
        self.top_model.train()
        true_ite, p_score = self.__get_labels(rnd, epoch)
        true_ite = true_ite.to(self.device)

        pred_ite = dcn_fp(epoch, middle_output, p_score, self.top_model)
        # y_f, y_cf = labels.chunk(2, 0)
        # true_ite = y_f - y_cf
        loss = self.criterion(pred_ite, true_ite)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_loss += loss.item()
        if rnd == 110 and epoch % 2 == 0:
            print("loss:", self.total_loss)
            self.total_loss = 0
        if rnd == 485 and epoch % 2 == 1:
            print("loss:", self.total_loss)
            self.total_loss = 0
        middle_grad = middle_output.grad

        return middle_grad

    def top_fp_bp(self, request, context):
        middle_output = pickle.loads(request.params_msg)

        rnd = request.round
        epoch = request.epoch
        middle_grad = self.__fp_bp(middle_output, rnd, epoch)

        response = label_owner_pb2.middle_grad(round=rnd,
                                               grad_msg=pickle.dumps(middle_grad))

        if epoch % 50 == 0 and rnd == 110:
            torch.save(self.top_model.state_dict(), "../save/dcn_pd/dcn_top_model_epoch{0}.pth".format(epoch))
            if epoch == 200:
                self.optimizer = optim.SGD(self.top_model.parameters(), lr=1e-5, momentum=0.9, nesterov=True,
                                           weight_decay=0.01)

        return response
