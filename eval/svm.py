import json
import os

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.get_train_test_data import train_test_mix_data, train_test_raw_data, train_test_synthetic_data


def SVMEval(name: str, idx=0):
    x_train, x_test, y_train, y_test = train_test_mix_data(name)

    svm_models1 = SVC()
    svm_models1 = svm_models1.fit(x_train, y_train)
    y_predict = svm_models1.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average="macro", zero_division=0)
    recall = recall_score(y_test, y_predict, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_predict, average="macro", zero_division=0)
    print("Mix：")
    print("SVM 准确率：{:.4f}".format(accuracy))
    print("SVM 精确率：{:.4f}".format(precision))
    print("SVM 召回率：{:.4f}".format(recall))
    print("SVM F1分数：{:.4f}".format(f1))

    result_dict = {
        "Dataset": name,
        "Method": "SVM",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    result_json = json.dumps(result_dict)

    if not os.path.exists(path="./eval/eval_result/SVM/mix/{}".format(idx)):
        os.mkdir("./eval/eval_result/SVM/mix/{}".format(idx))

    f = open("./eval/eval_result/SVM/mix/{}/{}.json".format(idx, name), 'w')
    f.write(result_json)

    x_train, x_test, y_train, y_test = train_test_raw_data(name)

    svm_models2 = SVC()
    svm_models2 = svm_models2.fit(x_train, y_train)
    y_predict = svm_models2.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average="macro", zero_division=0)
    recall = recall_score(y_test, y_predict, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_predict, average="macro", zero_division=0)
    print("Raw：")
    print("SVM 准确率：{:.4f}".format(accuracy))
    print("SVM 精确率：{:.4f}".format(precision))
    print("SVM 召回率：{:.4f}".format(recall))
    print("SVM F1分数：{:.4f}".format(f1))

    result_dict = {
        "Dataset": name,
        "Method": "SVM",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    result_json = json.dumps(result_dict)

    if not os.path.exists(path="./eval/eval_result/SVM/raw/{}".format(idx)):
        os.mkdir("./eval/eval_result/SVM/raw/{}".format(idx))

    f = open("./eval/eval_result/SVM/raw/{}/{}.json".format(idx, name), 'w')
    f.write(result_json)

    x_train, x_test, y_train, y_test = train_test_synthetic_data(name, idx)

    svm_models3 = SVC()
    svm_models3 = svm_models3.fit(x_train, y_train)
    y_predict = svm_models3.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average="macro", zero_division=0)
    recall = recall_score(y_test, y_predict, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_predict, average="macro", zero_division=0)
    print("Synthetic：")
    print("SVM 准确率：{:.4f}".format(accuracy))
    print("SVM 精确率：{:.4f}".format(precision))
    print("SVM 召回率：{:.4f}".format(recall))
    print("SVM F1分数：{:.4f}".format(f1))

    result_dict = {
        "Dataset": name,
        "Method": "SVM",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    result_json = json.dumps(result_dict)

    if not os.path.exists(path="./eval/eval_result/SVM/synthetic/{}".format(idx)):
        os.mkdir("./eval/eval_result/SVM/synthetic/{}".format(idx))

    f = open("./eval/eval_result/SVM/synthetic/{}/{}.json".format(idx, name), 'w')
    f.write(result_json)
