import json
import os

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

from eval.mlp_for_eval import train_model
from utils.get_train_test_data import train_test_raw_data


def smoteoversampling(name: str, idx=0):
    x_train, x_test, y_train, y_test = train_test_raw_data(name, idx=idx)
    labels = set(y_train)
    sampling_strategy = {}
    for label in labels:
        counts = list(y_train).count(label) * 100
        sampling_strategy[label] = counts

    ros = SMOTE(sampling_strategy=sampling_strategy)
    x_resample, y_resample = ros.fit_resample(x_train, y_train)
    return x_resample, y_resample, x_test, y_test


def evalsmote(y_test, y_predict):
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict, average="macro", zero_division=0)
    recall = recall_score(y_test, y_predict, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_predict, average="macro", zero_division=0)
    return accuracy, precision, recall, f1


def save_result(name, idx, method, accuracy, precision, recall, f1):
    result_dict = {
        "Dataset": name,
        "Method": method,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
    result_json = json.dumps(result_dict)
    if not os.path.exists(path="./eval/smote_result/{}/{}".format(method, idx)):
        os.mkdir("./eval/smote_result/{}/{}".format(method, idx))
    f = open("./eval/smote_result/{}/{}/{}.json".format(method, idx, name), 'w')
    f.write(result_json)


def smoteSVM(name: str, idx=0):
    method = "SVM"
    x_resample, y_resample, x_test, y_test = smoteoversampling(name, idx=idx)
    clf = SVC().fit(x_resample, y_resample)
    y_predict = clf.predict(x_test)
    accuracy, precision, recall, f1 = evalsmote(y_test, y_predict)
    print("SVM 准确率：{:.4f}".format(accuracy))
    print("SVM 精确率：{:.4f}".format(precision))
    print("SVM 召回率：{:.4f}".format(recall))
    print("SVM F1分数：{:.4f}".format(f1))
    save_result(name, idx, method, accuracy, precision, recall, f1)


def smoteLR(name: str, idx=0):
    method = "LogisticRegression"
    x_resample, y_resample, x_test, y_test = smoteoversampling(name, idx=idx)
    clf = LogisticRegression(max_iter=5000).fit(x_resample, y_resample)
    y_predict = clf.predict(x_test)
    accuracy, precision, recall, f1 = evalsmote(y_test, y_predict)
    print("Logistic Regression 准确率：{:.4f}".format(accuracy))
    print("Logistic Regression 精确率：{:.4f}".format(precision))
    print("Logistic Regression 召回率：{:.4f}".format(recall))
    print("Logistic Regression F1分数：{:.4f}".format(f1))
    save_result(name, idx, method, accuracy, precision, recall, f1)


def smoteMLP(name: str, idx=0):
    method = "MLP"
    x_resample, y_resample, x_test, y_test = smoteoversampling(name, idx=idx)
    accuracy, precision, recall, f1 = train_model(x_resample, x_test, y_resample, y_test)
    print("MLP 准确率：{:.4f}".format(accuracy))
    print("MLP 精确率：{:.4f}".format(precision))
    print("MLP 召回率：{:.4f}".format(recall))
    print("MLP F1分数：{:.4f}".format(f1))
    save_result(name, idx, method, accuracy, precision, recall, f1)


if __name__ == '__main__':
    for file in os.listdir("./result/raw_data"):
        a = "==================== Dataset: {} ====================".format(file)
        print(a)
        for count in range(10):
            print("第 {} 次留出法".format(count + 1))
            smoteSVM(file, count)
            print("-" * len(a))
            smoteLR(file, count)
            print("-" * len(a))
            smoteMLP(file, count)
            print("=" * len(a))
