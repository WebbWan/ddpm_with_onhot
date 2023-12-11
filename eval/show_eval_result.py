import json
import os

import numpy as np
import pandas as pd


def get_data(path: str):
    f = open(path, 'r')
    result_dict = json.load(f)
    return result_dict


def showEvalResult(method: str, style: str, counts: list, name: str, baseline=None):
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for count in counts:
        if not baseline:
            path = "./eval/eval_result/{}/{}/{}/{}.json".format(method, style, count, name)
        else:
            path = "./eval/{}_result/{}/{}/{}.json".format(baseline, method, count, name)
        result = get_data(path)
        accuracy_list.append(result["Accuracy"])
        precision_list.append(result["Precision"])
        recall_list.append(result["Recall"])
        f1_list.append(result["F1"])

    acc_numpy = np.array(accuracy_list)
    pre_numpy = np.array(precision_list)
    rec_numpy = np.array(recall_list)
    f1_numpy = np.array(f1_list)

    acc_mean = acc_numpy.mean()
    pre_mean = pre_numpy.mean()
    rec_mean = rec_numpy.mean()
    f1_mean = f1_numpy.mean()

    acc_std = acc_numpy.std()
    pre_std = pre_numpy.std()
    rec_std = rec_numpy.std()
    f1_std = f1_numpy.std()

    result_dict = {
        "Dataset": name,
        "Method": method,
        "Style": style,
        "Accuracy Mean": acc_mean,
        "Accuracy Std": acc_std,
        "Precision Mean": pre_mean,
        "Precision Std": pre_std,
        "Recall Mean": rec_mean,
        "Recall Std": rec_std,
        "F1 Mean": f1_mean,
        "F1 Std": f1_std
    }

    return result_dict


def showMethodResult(method: str, save_path: str, is_basline: bool, basline=None):
    result = []
    name_list = os.listdir("./data")
    count_list = [i for i in range(10)]
    for name in name_list:
        name = name.split(".")[0]
        if not is_basline:
            style_list1 = "mix"
            dict_res1 = showEvalResult(method, style_list1, count_list, name)
            result.append(pd.Series(dict_res1))

            style_list2 = "raw"
            dict_res2 = showEvalResult(method, style_list2, count_list, name)
            result.append(pd.Series(dict_res2))

            style_list3 = "synthetic"
            dict_res3 = showEvalResult(method, style_list3, count_list, name)
            result.append(pd.Series(dict_res3))
        else:
            dict_res1 = showEvalResult(method, "None", count_list, name, basline)
            result.append(pd.Series(dict_res1))

    pd.concat(result, axis=1).T.to_csv(save_path + "{}.csv".format(method), encoding="utf-8", index=False)
    print("{} result save successful !!!".format(method))
