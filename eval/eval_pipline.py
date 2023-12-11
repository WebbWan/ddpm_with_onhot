import os

from eval.svm import SVMEval
from eval.mlp_for_eval import MLPEval
from eval.logistic_regression import LogisticRegressionEval


def Pipline(count=10):
    for file in os.listdir("./data"):
        name = file.split(".")[0]
        a = "==================== Dataset: {} ====================".format(name)
        print(a)
        for c in range(count):
            b = "----------> 第 {} 次留出法 <----------".format(c + 1)
            print(b)
            LogisticRegressionEval(name, idx=c)
            print("-" * len(a))
            SVMEval(name, idx=c)
            print("-" * len(a))
            MLPEval(name, idx=c)
            print("=" * len(a))
