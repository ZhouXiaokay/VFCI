import torch.nn
from model.DragonNet import BottomModel
import torch.optim as optim
from torch.utils.data import DataLoader
from .base_trainer import BaseTrainer
from tqdm import tqdm


class DragonNetTrainer(BaseTrainer):
    def __init__(self, n_f, args, client_rank, dataset):
        super().__init__(n_f, args, client_rank, dataset)
        self.bottom_model = BottomModel(n_f)
        self.bottom_model.to(self.device)
        self.optimizer = optim.Adam(self.bottom_model.parameters(), lr=1e-3, weight_decay=0.01)
        self.optimizer_sgd = optim.SGD(self.bottom_model.parameters(), lr=1e-5, momentum=0.9, nesterov=True,
                                       weight_decay=0.01)

    def one_epoch_bottom_fp_bp(self, epoch):
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size,
                                                  shuffle=False, num_workers=0)
        rnd = 1
        for batch in data_loader:
            idx, train_x = batch
            train_x = train_x.to(self.device)
            bottom_output = self.bottom_model(train_x)
            grads = self.transmit(bottom_output, epoch, rnd)
            self.optimizer.zero_grad()
            bottom_output.backward(grads)
            self.optimizer.step()

            # print(rnd, idx)
            rnd += 1
        # print("bias",self.bottom_model.fc1.bias)

    def launch(self):
        epoch = self.args.epoch

        for epo in tqdm(range(1, epoch+1)):
            self.one_epoch_bottom_fp_bp(epo)
            if epo % 50 == 0:
                torch.save(self.bottom_model.state_dict(),
                           "../save/dragonnet/bottom_model_epoch{0}_client_{1}.pth".format(epo, self.client_rank))
                if epo == 201:
                    self.optimizer = optim.SGD(self.bottom_model.parameters(), lr=1e-5, momentum=0.9, nesterov=True,
                                               weight_decay=0.01)
