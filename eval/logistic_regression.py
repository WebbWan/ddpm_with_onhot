import json
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.get_train_test_data import train_test_mix_data, train_test_raw_data, train_test_synthetic_data


def LogisticRegressionEval(name: str, idx=0):
    x_train, x_test, y_train, y_test = train_test_mix_data(name)

    clf1 = LogisticRegression(max_iter=5000)
    clf1 = clf1.fit(x_train, y_train)
    y_predict = clf1.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average="macro", zero_division=0)
    recall = recall_score(y_test, y_predict, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_predict, average="macro", zero_division=0)
    print("Mix：")
    print("Logistic Regression 准确率：{:.4f}".format(accuracy))
    print("Logistic Regression 精确率：{:.4f}".format(precision))
    print("Logistic Regression 召回率：{:.4f}".format(recall))
    print("Logistic Regression F1分数：{:.4f}".format(f1))

    result_dict = {
        "Dataset": name,
        "Method": "Logistic Regression",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    result_json = json.dumps(result_dict)

    if not os.path.exists(path="./eval/eval_result/LogisticRegression/mix/{}".format(idx)):
        os.mkdir("./eval/eval_result/LogisticRegression/mix/{}".format(idx))

    f = open("./eval/eval_result/LogisticRegression/mix/{}/{}.json".format(idx, name), 'w')
    f.write(result_json)

    x_train, x_test, y_train, y_test = train_test_raw_data(name)

    clf2 = LogisticRegression(max_iter=5000)
    clf2 = clf2.fit(x_train, y_train)
    y_predict = clf2.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average="macro", zero_division=0)
    recall = recall_score(y_test, y_predict, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_predict, average="macro", zero_division=0)
    print("Raw：")
    print("Logistic Regression 准确率：{:.4f}".format(accuracy))
    print("Logistic Regression 精确率：{:.4f}".format(precision))
    print("Logistic Regression 召回率：{:.4f}".format(recall))
    print("Logistic Regression F1分数：{:.4f}".format(f1))

    result_dict = {
        "Dataset": name,
        "Method": "Logistic Regression",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    result_json = json.dumps(result_dict)

    if not os.path.exists(path="./eval/eval_result/LogisticRegression/raw/{}".format(idx)):
        os.mkdir("./eval/eval_result/LogisticRegression/raw/{}".format(idx))

    f = open("./eval/eval_result/LogisticRegression/raw/{}/{}.json".format(idx, name), 'w')
    f.write(result_json)

    x_train, x_test, y_train, y_test = train_test_synthetic_data(name, idx)

    clf3 = LogisticRegression(max_iter=5000)
    clf3 = clf3.fit(x_train, y_train)
    y_predict = clf3.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average="macro", zero_division=0)
    recall = recall_score(y_test, y_predict, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_predict, average="macro", zero_division=0)
    print("Synthetic：")
    print("Logistic Regression 准确率：{:.4f}".format(accuracy))
    print("Logistic Regression 精确率：{:.4f}".format(precision))
    print("Logistic Regression 召回率：{:.4f}".format(recall))
    print("Logistic Regression F1分数：{:.4f}".format(f1))

    result_dict = {
        "Dataset": name,
        "Method": "Logistic Regression",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    result_json = json.dumps(result_dict)

    if not os.path.exists(path="./eval/eval_result/LogisticRegression/synthetic/{}".format(idx)):
        os.mkdir("./eval/eval_result/LogisticRegression/synthetic/{}".format(idx))

    f = open("./eval/eval_result/LogisticRegression/synthetic/{}/{}.json".format(idx, name), 'w')
    f.write(result_json)
