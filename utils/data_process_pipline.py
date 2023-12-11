import json
import os.path

import numpy as np
import pandas as pd

from utils.data_transformer import restructuring, categorylabel, check_var_dataframe, get_weak_column, \
    tabular_data_extend, DataTransformer
from utils.get_train_test_data import get_ddpm_train_test_data


class DataPipline(object):
    def __init__(self):
        self.trans = None

    def process_pipline(self, path, num=0, idx=0):
        parent_path = "./data/"
        name = path.split(sep=".")[0]
        if not os.path.exists(path="./result/raw_data/{}".format(name)):
            os.mkdir("./result/raw_data/{}".format(name))

        if not os.path.exists(path="./result/raw_data/{}/{}".format(name, idx)):
            os.mkdir("./result/raw_data/{}/{}".format(name, idx))

        data_path = parent_path + path
        dataset_info = get_ddpm_train_test_data(path=data_path)

        con_list, cat_list = check_var_dataframe(dataset_info["train"].iloc[:, :-1])
        w_idx, w_value = get_weak_column(dataset_info["train"], len(con_list))
        print("Weak index: {}, MI values: {}".format(w_idx, w_value))
        mi_dict = {
            "index": w_idx,
            "values": w_value
        }
        mi_result_json = json.dumps(mi_dict)
        if not os.path.exists(path="./result/MI_info/{}".format(name)):
            os.mkdir("./result/MI_info/{}".format(name))
        f = open("./result/MI_info/{}/{}.json".format(name, idx), 'w')
        f.write(mi_result_json)

        dataset = tabular_data_extend(dataset_info["train"], num - 1)
        dataset, noise_class_list = categorylabel(dataset)

        self.trans = DataTransformer()
        self.trans.fit(dataset_info["raw"].iloc[:, :-1], cat_list)
        data_train_trans = self.trans.transform(dataset, w_idx, add_noise=True)
        data_train_trans_no_noise = self.trans.transform(dataset, w_idx, add_noise=False)
        data_test_trans = self.trans.transform(dataset_info["test"].iloc[:, :-1], w_idx, add_noise=False)

        if not os.path.exists(path="./result/train_data/{}".format(name)):
            os.mkdir("./result/train_data/{}".format(name))

        if not os.path.exists(path="./result/train_data/{}/{}".format(name, idx)):
            os.mkdir("./result/train_data/{}/{}".format(name, idx))

        train_data_concat = np.concatenate(
            (self.trans.transform(dataset_info["train"].iloc[:, :-1], w_idx, add_noise=False),
             dataset_info["train"].iloc[:, -1].to_numpy().reshape(-1, 1)), axis=1)
        np.save("./result/train_data/{}/{}/{}_train.npy".format(name, idx, name), train_data_concat)

        ra_st = 0
        ex_st = 0
        output = []
        num_list = []
        for raw, extend in zip(dataset_info["train_class"], noise_class_list):
            output.append(np.concatenate((data_train_trans[ex_st: ex_st + extend],
                                          data_train_trans_no_noise[ra_st: ra_st + raw])))
            num_list.append(raw + extend)
            ra_st += raw
            ex_st += extend

        data_output = np.concatenate(output, axis=0)

        st = 0
        for class_idx, cat_num in enumerate(dataset_info["test_class"]):
            class_idx_numpy = np.array(
                [class_idx] * cat_num
            ).reshape(-1, 1)
            data_class = np.concatenate((data_test_trans[st: st + cat_num, :], class_idx_numpy), axis=1)
            print("Class {} shape (raw): ".format(class_idx), data_class.shape)
            np.save("./result/raw_data/{}/{}/{}_raw_class{}_attr.npy".format(name, idx, name, class_idx),
                    data_class)
            st += cat_num

        return data_output, num_list

    def train_after_recover(self, data: np.ndarray):
        return self.trans.recover_onehot(data)
