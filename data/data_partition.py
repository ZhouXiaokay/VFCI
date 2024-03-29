import sys
import torch

sys.path.append("..")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataPartition:
    def __init__(self, csv_path, split_size, seed=3456, num_users=2):
        self.csv_path = csv_path
        self.split_size = split_size
        self.num_users = num_users
        self.seed = seed

    def vertical_split(self):
        df = pd.read_csv(self.csv_path, header=None)
        np_ID, np_covariates_X, np_treatment_Y, np_outcome_Y = self.__convert_to_numpy(df)
        num_features = int(np_covariates_X.shape[1] / self.num_users)
        all_idxs = [i for i in range(np_covariates_X.shape[1])]
        result = []
        for i in range(self.num_users):
            t = set(np.random.choice(all_idxs, num_features, replace=False))
            result.append(list(t))
            all_idxs = list(set(all_idxs) - t)
        i = 0
        for item in result:
            i += 1
            vertix = self.__concat_numpy(item, np_covariates_X)
            vertix = np.concatenate((np_ID, vertix), axis=1)
            if i == 2:
                vertix = np.concatenate((vertix, np_treatment_Y, np_outcome_Y), axis=1)
            vertidf = pd.DataFrame(vertix)
            vertidf.to_csv("client{0}.csv".format(i), header=False, index=False)

    def __concat_numpy(self, rlist, npx):
        nlist = []
        for i in rlist:
            if i == 24:
                nlist.append(npx[:, i:])
            else:
                nlist.append(npx[:, i:i + 1])

        return np.concatenate(tuple(nlist), axis=1)

    def preprocess_data_from_csv(self, csv_path, server_flag):
        print(".. Data Loading ..")
        # data load
        df = pd.read_csv(csv_path, header=None)

        if server_flag:
            np_ID = df.iloc[:, 0:1].to_numpy()
            covariates_X = df.iloc[:, 1:13].to_numpy()
            treatment_Y = df.iloc[:, 13:14].to_numpy()
            outcome_Y = df.iloc[:, 14:16].to_numpy()
            mu = df.iloc[:, 16:18].to_numpy()
            return np_ID, covariates_X, treatment_Y, outcome_Y, mu
        else:
            np_ID = df.iloc[:, 0:1].to_numpy()
            covariates_X = df.iloc[:, 1:14].to_numpy()
            return np_ID, covariates_X

    def getPSNTensor(self, csv_path, server_flag):
        if server_flag:
            np_ID, covariates_X, treatment_Y, outcome_Y, mu = self.preprocess_data_from_csv(csv_path, server_flag)
            id_train, id_test, x_train, x_test, t_y_train, t_y_test = train_test_split(np_ID, covariates_X, treatment_Y,
                                                                                       random_state=self.seed,
                                                                                       train_size=0.8)
            tensor_id_train = self.convert_to_tensor(id_train)
            tensor_id_test = self.convert_to_tensor(id_test)
            tensor_x_train = self.convert_to_tensor(x_train)
            tensor_x_test = self.convert_to_tensor(x_test)
            tensor_t_train = self.convert_to_tensor(t_y_train)
            tensor_t_test = self.convert_to_tensor(t_y_test)
            # return tensor_id_train, tensor_id_test, tensor_x_train, tensor_x_test, t_y_train, t_y_test
            return {"PSN_trainData": (tensor_id_train, tensor_x_train, tensor_t_train),
                    "PSN_testData": (tensor_id_test, tensor_x_test, tensor_t_test)}
        else:
            np_ID, covariates_X = self.preprocess_data_from_csv(csv_path, server_flag)
            id_train, id_test, x_train, x_test = train_test_split(np_ID, covariates_X,
                                                                  random_state=self.seed,
                                                                  train_size=0.8)
            tensor_id_train = self.convert_to_tensor(id_train)
            tensor_id_test = self.convert_to_tensor(id_test)
            tensor_x_train = self.convert_to_tensor(x_train)
            tensor_x_test = self.convert_to_tensor(x_test)

            return {"PSN_trainData": (tensor_id_train, tensor_x_train),
                    "PSN_testData": (tensor_id_test, tensor_x_test)}

    """
    input:csv,severflag
    output:dict:train set and test set
           0:id 1:x 2:outcome and treatment 
    """

    def getDragonNetTensor(self, csv_path, server_flag):
        if server_flag:
            np_ID, covariates_X, treatment_Y, outcome_Y, mu = self.preprocess_data_from_csv(csv_path, server_flag)
            yt = np.concatenate([outcome_Y, treatment_Y], 1)
            id_train, id_test, x_train, x_test, yt_train, yt_test, mu_train, mu_test = train_test_split(np_ID,
                                                                                                        covariates_X,
                                                                                                        yt, mu,
                                                                                                        random_state=self.seed,
                                                                                                        train_size=0.8)
            tensor_id_train = self.convert_to_tensor(id_train)
            tensor_id_test = self.convert_to_tensor(id_test)
            tensor_x_train = self.convert_to_tensor(x_train)
            tensor_x_test = self.convert_to_tensor(x_test)
            tensor_yt_train = self.convert_to_tensor(yt_train)
            tensor_yt_test = self.convert_to_tensor(yt_test)
            tensor_mu_train = self.convert_to_tensor(mu_train)
            tensor_mu_test = self.convert_to_tensor(mu_test)
            # return tensor_id_train, tensor_id_test, tensor_x_train, tensor_x_test, t_y_train, t_y_test
            return {"DragonNet_trainData": (tensor_id_train, tensor_x_train, tensor_yt_train, tensor_mu_train),
                    "DragonNet_testData": (tensor_id_test, tensor_x_test, tensor_yt_test, tensor_mu_test)}
        else:
            np_ID, covariates_X = self.preprocess_data_from_csv(csv_path, server_flag)
            id_train, id_test, x_train, x_test = train_test_split(np_ID, covariates_X,
                                                                  random_state=self.seed,
                                                                  train_size=0.8)
            tensor_id_train = self.convert_to_tensor(id_train)
            tensor_id_test = self.convert_to_tensor(id_test)
            tensor_x_train = self.convert_to_tensor(x_train)
            tensor_x_test = self.convert_to_tensor(x_test)

            return {"DragonNet_trainData": (tensor_id_train, tensor_x_train),
                    "DragonNet_testData": (tensor_id_test, tensor_x_test)}

    def convert_to_tensor(self, np):
        tensor = torch.from_numpy(np).float()
        return tensor

    def preDCNData(self, csv_path, treat):
        df = pd.read_csv(csv_path, header=None)
        df = df[df.iloc[:, -5] == treat]

        np_ID = df.iloc[:, 0:1].to_numpy()
        covariates_X = df.iloc[:, 1:13].to_numpy()
        treatment_Y = df.iloc[:, 13:14].to_numpy()
        outcome_Y = df.iloc[:, 14:16].to_numpy()
        mu = df.iloc[:, 16:18].to_numpy()
        return np_ID, covariates_X, treatment_Y, outcome_Y, mu

    """
    :input:csv,index
    :return:dict:train set and test set
           0:id 1:x 
    """

    def getDCNTensorWithLabel(self, csv_path, phase):

        # treated data load
        treat_np_ID, treat_covariates_X, treat_treatment_Y, treat_outcome_Y, treat_mu = \
            self.preDCNData(csv_path, treat=1)
        treat_id_train, treat_id_test, treat_x_train, treat_x_test, treat_treat_train, treat_treat_test, \
        treat_out_y_train, treat_out_y_test, treat_mu_train, treat_mu_test = train_test_split(treat_np_ID,
                                                                                              treat_covariates_X,
                                                                                              treat_treatment_Y,
                                                                                              treat_outcome_Y,
                                                                                              treat_mu,
                                                                                              random_state=self.seed,
                                                                                              train_size=0.8)

        treat_tensor_id_train = self.convert_to_tensor(treat_id_train)
        treat_tensor_id_test = self.convert_to_tensor(treat_id_test)
        treat_tensor_x_train = self.convert_to_tensor(treat_x_train)
        treat_tensor_x_test = self.convert_to_tensor(treat_x_test)
        treat_tensor_treat_train = self.convert_to_tensor(treat_treat_train)
        treat_tensor_treat_test = self.convert_to_tensor(treat_treat_test)
        treat_tensor_outcome_train = self.convert_to_tensor(treat_out_y_train)
        treat_tensor_outcome_test = self.convert_to_tensor(treat_out_y_test)
        treat_tensor_mu_train = self.convert_to_tensor(treat_mu_train)
        treat_tensor_mu_test = self.convert_to_tensor(treat_mu_test)

        # control data load
        control_np_ID, control_covariates_X, control_treatment_Y, control_outcome_Y, control_mu = \
            self.preDCNData(csv_path, treat=0)

        control_id_train, control_id_test, control_x_train, control_x_test, control_treat_train, control_treat_test, \
        control_out_y_train, control_out_y_test, control_mu_train, control_mu_test = train_test_split(control_np_ID,
                                                                                                      control_covariates_X,
                                                                                                      control_treatment_Y,
                                                                                                      control_outcome_Y,
                                                                                                      control_mu,
                                                                                                      random_state=self.seed,
                                                                                                      train_size=0.8)

        control_tensor_id_train = self.convert_to_tensor(control_id_train)
        control_tensor_id_test = self.convert_to_tensor(control_id_test)
        control_tensor_x_train = self.convert_to_tensor(control_x_train)
        control_tensor_x_test = self.convert_to_tensor(control_x_test)
        control_tensor_treat_train = self.convert_to_tensor(control_treat_train)
        control_tensor_treat_test = self.convert_to_tensor(control_treat_test)
        control_tensor_outcome_train = self.convert_to_tensor(control_out_y_train)
        control_tensor_outcome_test = self.convert_to_tensor(control_out_y_test)
        control_tensor_mu_train = self.convert_to_tensor(control_mu_train)
        control_tensor_mu_test = self.convert_to_tensor(control_mu_test)

        # accord phase return train set or test set
        if phase == "train":
            return {"treat_trainData": (
                treat_tensor_id_train, treat_tensor_x_train, treat_tensor_treat_train, treat_tensor_outcome_train,
                treat_tensor_mu_train),
                "control_trainData": (
                    control_tensor_id_train, control_tensor_x_train, control_tensor_treat_train,
                    control_tensor_outcome_train, control_tensor_mu_train)}
        if phase == "test":
            return {"treat_testData": (
                treat_tensor_id_test, treat_tensor_x_test, treat_tensor_treat_test, treat_tensor_outcome_test,
                treat_tensor_mu_test),
                "control_testData": (
                    control_tensor_id_test, control_tensor_x_test, control_tensor_treat_test,
                    control_tensor_outcome_test, control_tensor_mu_test)}

    """
    input:csv,index
    output:dict:train set and test set
           0:id 1:x 
    """

    def getIndex(self, csv_path):
        df = pd.read_csv(csv_path, header=None)
        treat_df = df[df.iloc[:, -5] == 1]
        control_df = df[df.iloc[:, -5] == 0]
        treat_df_id = treat_df.iloc[:, 0]
        control_df_id = control_df.iloc[:, 0]
        return {"treat": treat_df_id.tolist(), "control": control_df_id.tolist()}

    def getDCNTensorWithoutLabel(self, csv_path, phase, index_dict):

        # treated data load

        # df = pd.read_csv(csv_path, header=None, error_bad_lines=False)
        df = pd.read_csv(csv_path, header=None)
        treat_df = df[df.iloc[:, 0].isin(index_dict["treat"])]

        treat_np_ID = treat_df.iloc[:, 0:1].to_numpy()
        treat_covariates_X = treat_df.iloc[:, 1:14].to_numpy()
        treat_id_train, treat_id_test, treat_x_train, treat_x_test = train_test_split(treat_np_ID, treat_covariates_X,
                                                                                      random_state=self.seed,
                                                                                      train_size=0.8)

        treat_tensor_id_train = self.convert_to_tensor(treat_id_train)
        treat_tensor_id_test = self.convert_to_tensor(treat_id_test)
        treat_tensor_x_train = self.convert_to_tensor(treat_x_train)
        treat_tensor_x_test = self.convert_to_tensor(treat_x_test)

        # control data load
        control_df = df[df.iloc[:, 0].isin(index_dict["control"])]

        control_np_ID = control_df.iloc[:, 0:1].to_numpy()
        control_covariates_X = control_df.iloc[:, 1:14].to_numpy()

        control_id_train, control_id_test, control_x_train, control_x_test = train_test_split(control_np_ID,
                                                                                              control_covariates_X,
                                                                                              random_state=self.seed,
                                                                                              train_size=0.8)

        control_tensor_id_train = self.convert_to_tensor(control_id_train)
        control_tensor_id_test = self.convert_to_tensor(control_id_test)
        control_tensor_x_train = self.convert_to_tensor(control_x_train)
        control_tensor_x_test = self.convert_to_tensor(control_x_test)

        # accord phase return train set or test set
        if phase == "train":
            return {"treat_trainData": (
                treat_tensor_id_train, treat_tensor_x_train),
                "control_trainData": (
                    control_tensor_id_train, control_tensor_x_train)}
        if phase == "test":
            return {"treat_testData": (
                treat_tensor_id_test, treat_tensor_x_test),
                "control_testData": (
                    control_tensor_id_test, control_tensor_x_test)}

#
# if __name__ == "__main__":
#
#     csv_path = './Dataset/ihdp_new.csv'
#     dp = DataPartition(csv_path, 0.8, 2)
#     dp.verticalParti()
#     id, x, t, o = dp.preprocess_data_from_csv('client2.csv', True)
#     dic = dp.getDCNTensorWithLabel('client2.csv', "train")
#     a,b=dic["treat_trainData"][3][0].chunk(2,0)
#     print(a,b)
#     index = dp.getIndex('client2.csv')
#     dic1=dp.getDCNTensorWithoutLabel('client1.csv',"train",index)
#     print(dic1["control_trainData"][0])
