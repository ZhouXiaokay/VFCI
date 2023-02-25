import torch.nn
from model.DCN import BottomModel
import torch.optim as optim
from .base_trainer import BaseTrainer
from tqdm import tqdm


class DCNTrainer(BaseTrainer):
    def __init__(self, n_f, args, client_rank, dataset):
        super().__init__(n_f, args, client_rank, dataset)
        self.bottom_model = BottomModel(n_f).to(self.device)
        self.optimizer = optim.Adam(self.bottom_model.parameters(), lr=1e-3)

    def one_epoch_bottom_fp_bp(self, epoch):
        treated_idx = self.dataset["treated"][0]
        control_idx = self.dataset["control"][0]
        treated_x = self.dataset["treated"][1]
        control_x = self.dataset["control"][1]

        if epoch % 2 == 0:
            # train treat data
            for rnd in range(len(treated_x)):
                # print("id", treated_idx[rnd])
                train_x = treated_x[rnd]
                train_x = train_x.to(self.device)
                bottom_output = self.bottom_model(train_x)
                grads = self.transmit(bottom_output, epoch, rnd)
                self.optimizer.zero_grad()
                bottom_output.backward(grads)
                self.optimizer.step()
        elif epoch % 2 == 1:
            for rnd in range(len(control_x)):
                # print("id", control_idx[rnd])
                train_x = control_x[rnd]
                train_x = train_x.to(self.device)
                bottom_output = self.bottom_model(train_x)
                grads = self.transmit(bottom_output, epoch, rnd)
                self.optimizer.zero_grad()
                bottom_output.backward(grads)
                self.optimizer.step()

    def launch(self):
        epoch = self.args.epoch

        for epo in tqdm(range(1, epoch + 1)):
            self.one_epoch_bottom_fp_bp(epo)

        torch.save(self.bottom_model.state_dict(),
                   "../save/dcn_pd/dcn_bottom_model_epoch{0}_client_{1}.pth".format(epoch, self.client_rank))
