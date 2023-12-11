import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.get_train_test_data import train_test_mix_data, train_test_raw_data, train_test_synthetic_data


class MlpForEval(nn.Module):
    class Block(nn.Module):

        def __init__(self, d_in, d_out, bias, dropout):
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
            self,
            d_in,
            d_layers,
            dropouts,
            d_out,
    ):
        super().__init__()

        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)

        self.blocks = nn.ModuleList(
            [
                MlpForEval.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )

        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x):
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x.squeeze()


class EvalDataset(Dataset):

    def __init__(self, dataset_attr, dataset_label):
        super().__init__()
        self.dataset_attr = dataset_attr
        self.dataset_label = dataset_label

    def __getitem__(self, item):
        d_a = self.dataset_attr[item]
        d_l = self.dataset_label[item]

        return d_a, d_l

    def __len__(self):
        return len(self.dataset_attr)


def train_model(
        x_train,
        x_test,
        y_train,
        y_test,
        epochs=200,
        batch_size=64,
        hidden_unit=256,
        deep=3,
        dropouts=0.2,
        learning_rate=1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_out = len(pd.DataFrame(np.concatenate((y_train, y_test))).value_counts())
    d_layer = [hidden_unit] * deep

    train_dataset = EvalDataset(x_train, y_train)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = MlpForEval(
        d_in=x_train.shape[1],
        d_out=d_out,
        d_layers=d_layer,
        dropouts=dropouts
    ).to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        loss_res = []
        for idx, (data, label) in enumerate(train_dataloader):
            data = data.to(device)
            label = label.long().to(device)
            output = model(data)
            if len(output.shape) != 2:
                output = output.reshape(1, -1)
            loss = loss_fn(output, label)
            loss_res.append(loss.cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_predict = model(torch.tensor(x_test, device=device)).argmax(1).cpu().numpy()
        y_test = y_test.astype(int)
        accuracy = accuracy_score(y_test, y_predict)
        precision = precision_score(y_test, y_predict, average="macro", zero_division=0)
        recall = recall_score(y_test, y_predict, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_predict, average="macro", zero_division=0)

    return accuracy, precision, recall, f1


def MLPEval(name: str, idx=0):
    x_train, x_test, y_train, y_test = train_test_mix_data(name, idx=idx)

    accuracy, precision, recall, f1 = train_model(
        x_train,
        x_test,
        y_train,
        y_test
    )
    print("Mix：")
    print("MLP 准确率：{:.4f}".format(accuracy))
    print("MLP 精确率：{:.4f}".format(precision))
    print("MLP 召回率：{:.4f}".format(recall))
    print("MLP F1分数：{:.4f}".format(f1))

    result_dict = {
        "Dataset": name,
        "Method": "MLP",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    result_json = json.dumps(result_dict)

    if not os.path.exists(path="./eval/eval_result/MLP/mix/{}".format(idx)):
        os.mkdir("./eval/eval_result/MLP/mix/{}".format(idx))

    f = open("./eval/eval_result/MLP/mix/{}/{}.json".format(idx, name), 'w')
    f.write(result_json)

    x_train, x_test, y_train, y_test = train_test_raw_data(name, idx=idx)

    accuracy, precision, recall, f1 = train_model(
        x_train,
        x_test,
        y_train,
        y_test
    )
    print("Raw：")
    print("MLP 准确率：{:.4f}".format(accuracy))
    print("MLP 精确率：{:.4f}".format(precision))
    print("MLP 召回率：{:.4f}".format(recall))
    print("MLP F1分数：{:.4f}".format(f1))

    result_dict = {
        "Dataset": name,
        "Method": "MLP",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    result_json = json.dumps(result_dict)

    if not os.path.exists(path="./eval/eval_result/MLP/raw/{}".format(idx)):
        os.mkdir("./eval/eval_result/MLP/raw/{}".format(idx))

    f = open("./eval/eval_result/MLP/raw/{}/{}.json".format(idx, name), 'w')
    f.write(result_json)

    x_train, x_test, y_train, y_test = train_test_synthetic_data(name, idx=idx)

    accuracy, precision, recall, f1 = train_model(
        x_train,
        x_test,
        y_train,
        y_test
    )
    print("Synthetic：")
    print("MLP 准确率：{:.4f}".format(accuracy))
    print("MLP 精确率：{:.4f}".format(precision))
    print("MLP 召回率：{:.4f}".format(recall))
    print("MLP F1分数：{:.4f}".format(f1))

    result_dict = {
        "Dataset": name,
        "Method": "MLP",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    result_json = json.dumps(result_dict)

    if not os.path.exists(path="./eval/eval_result/MLP/synthetic/{}".format(idx)):
        os.mkdir("./eval/eval_result/MLP/synthetic/{}".format(idx))

    f = open("./eval/eval_result/MLP/synthetic/{}/{}.json".format(idx, name), 'w')
    f.write(result_json)
