import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.data_transformer import restructuring, categorylabel


def train_test_mix_data(name: str, idx=0):
    # 读取数据集
    raw_data_parent_path = "./result/raw_data/"
    synthetic_data_parent_path = "./result/synthetic_data/"

    data_raw = []
    for file in os.listdir(raw_data_parent_path + name + "/{}/".format(idx)):
        data_class = np.load(raw_data_parent_path + name + "/{}/".format(idx) + file)
        data_raw.append(data_class)

    data_raw_numpy = np.concatenate(data_raw, axis=0)
    x_raw = data_raw_numpy[:, :-1]
    y_raw = data_raw_numpy[:, -1].astype(int)

    data_synthetic = []
    for file in os.listdir(synthetic_data_parent_path + name + "/{}/".format(idx)):
        data_class = np.load(synthetic_data_parent_path + name + "/{}/".format(idx) + file)
        data_synthetic.append(data_class)

    data_synthetic_numpy = np.concatenate(data_synthetic, axis=0)

    x_synthetic = data_synthetic_numpy[:, :-1]
    y_synthetic = data_synthetic_numpy[:, -1].astype(int)

    # 划分训练集测试集
    x_raw_train, x_raw_test, y_raw_train, y_raw_test = train_test_split(
        x_raw, y_raw,
        train_size=0.2,
        shuffle=True,
        stratify=y_raw
    )

    # x_synthetic_train, x_synthetic_test, y_synthetic_train, y_synthetic_test = train_test_split(
    #     x_synthetic, y_synthetic,
    #     train_size=0.5,
    #     shuffle=True,
    #     stratify=y_synthetic
    # )

    x_train = np.concatenate((x_synthetic, x_raw_train), axis=0)
    y_train = np.concatenate((y_synthetic, y_raw_train), axis=0)
    x_test = x_raw_test
    y_test = y_raw_test

    return x_train, x_test, y_train, y_test


def train_test_raw_data(name: str, idx=0):
    raw_data_parent_path = "./result/raw_data/"

    data_raw = []
    for file in os.listdir(raw_data_parent_path + name + "/{}/".format(idx)):
        data_class = np.load(raw_data_parent_path + name + "/{}/".format(idx) + file)
        data_raw.append(data_class)

    data_raw_numpy = np.concatenate(data_raw, axis=0)
    x_raw_test = data_raw_numpy[:, :-1]
    y_raw_test = data_raw_numpy[:, -1].astype(int)

    train_data_parent_path = "./result/train_data/{}/{}/{}_train.npy".format(name, idx, name)
    data_train = np.load(train_data_parent_path)
    np.random.shuffle(data_train)

    x_raw_train = data_train[:, :-1]
    y_raw_train = data_train[:, -1].astype(int)

    return x_raw_train, x_raw_test, y_raw_train, y_raw_test


def train_test_synthetic_data(name: str, idx=0):
    # 读取数据集
    raw_data_parent_path = "./result/raw_data/"

    data_raw = []
    for file in os.listdir(raw_data_parent_path + name + "/{}/".format(idx)):
        data_class = np.load(raw_data_parent_path + name + "/{}/".format(idx) + file)
        data_raw.append(data_class)

    data_raw_numpy = np.concatenate(data_raw, axis=0)
    x_raw = data_raw_numpy[:, :-1]
    y_raw = data_raw_numpy[:, -1].astype(int)

    synthetic_data_parent_path = "./result/synthetic_data/"

    data_synthetic = []
    for file in os.listdir(synthetic_data_parent_path + name + "/{}/".format(idx)):
        data_class = np.load(synthetic_data_parent_path + name + "/{}/".format(idx) + file)
        data_synthetic.append(data_class)

    data_synthetic_numpy = np.concatenate(data_synthetic, axis=0)
    np.random.shuffle(data_synthetic_numpy)
    x_synthetic = data_synthetic_numpy[:, :-1]
    y_synthetic = data_synthetic_numpy[:, -1].astype(int)

    return x_synthetic, x_raw, y_synthetic, y_raw


def get_ddpm_train_test_data(path: str, alpha=5):
    dataset_raw = pd.read_csv(path, header=None)
    dataset_raw = restructuring(dataset_raw)
    dataset_raw, raw_class_list = categorylabel(dataset_raw)

    attr_num = dataset_raw.shape[-1] - 1

    st = 0
    train_data = []
    test_data = []
    train_class_list = []
    test_class_list = []

    for class_num in raw_class_list:
        dataset_class = dataset_raw.iloc[st:st + class_num, :].sample(frac=1).reset_index(drop=True)

        if class_num <= alpha * attr_num:
            sample_num = int(0.66 * class_num)
        else:
            sample_num = int(alpha * attr_num)

        train_data.append(dataset_class.iloc[:sample_num, :])
        test_data.append(dataset_class.iloc[sample_num:, :])
        train_class_list.append(sample_num)
        test_class_list.append(class_num - sample_num)
        st += class_num

    train_set = pd.concat(train_data, ignore_index=True)
    test_set = pd.concat(test_data, ignore_index=True)

    return {
        "raw": dataset_raw,
        "train": train_set,
        "test": test_set,
        "train_class": train_class_list,
        "test_class": test_class_list
    }
