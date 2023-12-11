import os

import torch
import random
import numpy as np

from eval.eval_pipline import Pipline
from models.model import Model
from utils.data_process_pipline import DataPipline


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(count=10):
    seed_torch()
    for file in os.listdir("./data"):
        path = file
        name = file.split(sep=".")[0]
        print("{}: ".format(name))
        extend_num = 100
        for c in range(count):
            print("第{}次留出法".format(c + 1))
            dp = DataPipline()
            train_data, classes_num = dp.process_pipline(path, extend_num, c)

            dim = train_data.shape[1]
            lr = 1e-3
            wd = 1e-5
            num_class = len(classes_num)
            deep = 3
            hidden_unit = 128
            dim_t = 32
            epochs = 1000
            batch_size = 128
            num_timesteps = 1500
            dropouts = 0.1

            d_layer = [hidden_unit] * deep

            if not os.path.exists(path="./result/synthetic_data/{}".format(name)):
                os.mkdir("./result/synthetic_data/{}".format(name))

            if not os.path.exists(path="./result/synthetic_data/{}/{}".format(name, c)):
                os.mkdir("./result/synthetic_data/{}/{}".format(name, c))

            std = 0
            for idx, class_num in enumerate(classes_num):
                print("Class {} training ...".format(idx))
                model = Model(
                    dim=dim,
                    lr=lr,
                    wd=wd,
                    num_class=num_class,
                    d_layer=d_layer,
                    dim_t=dim_t,
                    epochs=epochs,
                    batch_size=batch_size,
                    num_timesteps=num_timesteps,
                    dropouts=dropouts
                )
                model.train_loop(torch.tensor(train_data[std:std + class_num, :]))
                output = dp.train_after_recover(model.samples_fn(class_num))

                class_idx_numpy = np.array(
                    [idx] * class_num
                ).reshape(-1, 1)

                data_result = np.concatenate((output, class_idx_numpy), axis=1)
                print("Class {} shape (synthetic): ".format(idx), data_result.shape)
                np.save("./result/synthetic_data/{}/{}/{}_syn_class{}_attr.npy".format(name, c, name, idx), data_result)
                std += class_num


if __name__ == '__main__':
    num = 10
    # main(count=num)
    Pipline(count=num)

