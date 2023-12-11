import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':
    for file in os.listdir("../data"):
        name = file.split(sep=".")[0]
        if not os.path.exists(path="./sample_target_img/{}".format(name)):
            os.mkdir("./sample_target_img/{}".format(name))
        for idx in range(len(os.listdir("./raw_data/{}".format(name)))):
            target_path = "./raw_data/{}/{}_raw_class{}_attr.npy".format(name, name, idx)
            sample_path = "./synthetic_data/{}/{}_syn_class{}_attr.npy".format(name, name, idx)

            # sample = pd.DataFrame(np.load(sample_path))
            target = pd.DataFrame(np.load(target_path)).iloc[:, :-1]
            sample = pd.DataFrame(np.load(sample_path)).sample(target.shape[0]).iloc[:, :-1]

            print("Sample shape: ", sample.shape)
            print("Target shape: ", target.shape)

            sample["test_type"] = "sample_{}".format(sample.shape[0])
            target["test_type"] = "target_{}".format(target.shape[0])
            result_cat = pd.concat((sample, target), axis=0)
            plt.figure(figsize=(14, 14))
            fig = sns.pairplot(result_cat, hue="test_type", palette="husl")
            plt.show()
            fig.savefig("./sample_target_img/{}/{}_class{}.png".format(name, name, idx))
