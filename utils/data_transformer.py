import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import namedtuple
from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import QuantileTransformer, MultiLabelBinarizer


class DataTransformer(object):
    def __init__(self):
        self.output_dimensions = None
        self.column_transform_info_list = None
        self.result_dict = namedtuple(
            "result_dict", ["column_type", "transformer", "output_dim"]
        )

    def _fit_continuous(self, data: pd.DataFrame) -> namedtuple:
        scaler = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(data.shape[0] // 30, 1000), 10),
            subsample=1000000000,
        )
        scaler.fit(data.to_numpy().reshape(-1, 1))

        return self.result_dict(
            column_type="continuous",
            transformer=scaler,
            output_dim=1
        )

    def _fit_discrete(self, data: pd.DataFrame) -> namedtuple:
        data_cat = data.values.reshape(-1, 1)
        ohe = MultiLabelBinarizer()
        ohe.fit(data_cat)
        num_categories = len(data.value_counts())

        return self.result_dict(
            column_type="discrete",
            transformer=ohe,
            output_dim=num_categories
        )

    def fit(self, raw_data: pd.DataFrame, discrete_columns_index=()) -> None:
        self.column_transform_info_list = []
        self.output_dimensions = 0

        for column in range(raw_data.shape[1]):

            if column in discrete_columns_index:
                column_transform_info = self._fit_discrete(raw_data.iloc[:, column])
            else:
                column_transform_info = self._fit_continuous(raw_data.iloc[:, column])
            self.column_transform_info_list.append(column_transform_info)
            self.output_dimensions += column_transform_info.output_dim

    @staticmethod
    def _transform_continuous(column_transform_info, data, weak=False):
        scaler = column_transform_info.transformer
        data_encoder = scaler.transform(data.to_numpy().reshape(-1, 1))
        if weak:
            noise = np.random.normal(loc=0, scale=0.01, size=data_encoder.shape)
            data_encoder = data_encoder + noise
        return data_encoder

    @staticmethod
    def _transform_discrete(column_transform_info, data: pd.DataFrame, weak=False):
        ohe = column_transform_info.transformer
        data_encoder = ohe.transform(data.values.reshape(-1, 1))
        if weak:
            noise = np.random.normal(loc=0, scale=0.01, size=data_encoder.shape)
            data_encoder = data_encoder + noise
        return data_encoder

    def _synchronous_transform(self, raw_data, column_transform_info_list, weak_idx, add_noise):
        column_data_list = []
        weak_col = False
        for idx, column_transform_info in enumerate(column_transform_info_list):
            if idx == weak_idx and add_noise:
                weak_col = True
            data = raw_data.iloc[:, idx]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data, weak_col))
            else:
                column_data_list.append(self._transform_discrete(column_transform_info, data, weak_col))

        return column_data_list

    def _parallel_transform(self, raw_data, column_transform_info_list, weak_idx, add_noise):
        processes = []
        weak_col = False
        for idx, column_transform_info in enumerate(column_transform_info_list):
            if idx == weak_idx and add_noise:
                weak_col = True
            data = raw_data.iloc[:, idx]
            if column_transform_info.column_type == 'continuous':
                process = delayed(self._transform_continuous)(column_transform_info, data, weak_col)
            else:
                process = delayed(self._transform_discrete)(column_transform_info, data, weak_col)
            processes.append(process)

        return Parallel(n_jobs=1)(processes)

    def transform(self, raw_data, weak_idx, add_noise=True):
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)

        if raw_data.shape[0] < 500:
            column_data_list = self._synchronous_transform(
                raw_data,
                self.column_transform_info_list,
                weak_idx,
                add_noise,
            )
        else:
            column_data_list = self._parallel_transform(
                raw_data,
                self.column_transform_info_list,
                weak_idx,
                add_noise,
            )
        output = np.concatenate(column_data_list, axis=1)
        return output

    def recover_onehot(self, data: np.ndarray):
        count = 0
        output = np.zeros(shape=data.shape)
        for idx, column_transform_info in enumerate(self.column_transform_info_list):
            if column_transform_info.column_type == "continuous":
                output[:, count:count+1] = data[:, count:count+1]
                # output[:, count:count + 1] = column_transform_info.transformer.inverse_transform(
                #     data[:, count:count + 1].reshape(-1, 1))
                count += 1
            else:
                cat_softmax = F.softmax(
                    torch.from_numpy(data[:, count: count + column_transform_info.output_dim]),
                    dim=1,
                ).numpy()
                onehot_encoded = np.zeros_like(cat_softmax)
                onehot_encoded[np.arange(len(cat_softmax)), cat_softmax.argmax(1)] = 1
                output[:, count: count + column_transform_info.output_dim] = onehot_encoded
                count += column_transform_info.output_dim

        return output


def check_var_dataframe(dataframe):
    # 初始化离散属性名列表和连续属性名列表
    discrete_attributes = []
    continuous_attributes = []

    # 遍历DataFrame的列
    for column in dataframe.columns:
        if np.issubdtype(dataframe[column].dtype, np.number):
            # 如果列的数据类型是数字，将其视为连续属性
            continuous_attributes.append(column)
        else:
            # 否则，将其视为离散属性
            discrete_attributes.append(column)

    return continuous_attributes, discrete_attributes


def restructuring(dataframe):
    dataframe_attr = dataframe.iloc[:, :-1]
    dataframe_label = dataframe.iloc[:, -1]
    con_var_list, cat_var_list = check_var_dataframe(dataframe_attr)
    dataframe_re_con = pd.DataFrame()
    if len(con_var_list) != 0:

        for i in con_var_list:
            dataframe_re_con = pd.concat((dataframe_re_con, dataframe_attr.iloc[:, i]), axis=1, ignore_index=True)

    dataframe_re_cat = pd.DataFrame()
    if len(cat_var_list) != 0:
        for i in cat_var_list:
            dataframe_re_cat = pd.concat((dataframe_re_cat, dataframe_attr.iloc[:, i]), axis=1, ignore_index=True)

    dataframe_re = pd.concat((dataframe_re_con, dataframe_re_cat), axis=1, ignore_index=True)
    dataframe_re = pd.concat((dataframe_re, dataframe_label), axis=1, ignore_index=True)
    return dataframe_re


def categorylabel(dataframe):
    dataframe_label = dataframe.iloc[:, -1]
    label_list = dataframe_label.unique()
    label_dict = {}

    for i in range(len(label_list)):
        label_dict[label_list[i]] = i

    dataframe[dataframe.columns[-1]] = dataframe[dataframe.columns[-1]].map(label_dict)

    dataframe_new = pd.DataFrame()
    classes_num = []
    for i in range(len(label_list)):
        data_class = dataframe[dataframe.iloc[:, -1] == i]
        num = data_class.shape[0]

        dataframe_new = pd.concat((dataframe_new, data_class), axis=0, ignore_index=True)
        classes_num.append(num)

    return dataframe_new, classes_num


def get_weak_column(data, con_length):
    data_attr = data.iloc[:, :-1]
    data_label = data.iloc[:, -1]

    mi_result_con = None
    if con_length != 0:
        data_temp_con = data_attr.iloc[:, :con_length]
        mi_result_con = mutual_info_regression(data_temp_con, data_label, random_state=40).tolist()

    mi_result_cat = []
    if data_attr.shape[1] - con_length != 0:
        cat_length = data_attr.shape[1] - con_length
        data_temp_cat = data_attr.iloc[:, con_length:]

        for i in range(cat_length):
            mi_result_cat.append(mutual_info_score(data_temp_cat.iloc[:, i], data_label))

    if not mi_result_con:
        mi_result = mi_result_cat
    elif not mi_result_cat:
        mi_result = mi_result_con
    else:
        mi_result = mi_result_con + mi_result_cat
        
    weak_index, weak_mi_value = min_value_index(mi_result)
    return weak_index, weak_mi_value


def min_value_index(ndarray):
    min_index = 0
    min_value = 0
    for i in range(len(ndarray)):
        if ndarray[i] <= ndarray[min_index]:
            min_index = i
            min_value = ndarray[i]
    return min_index, min_value


def tabular_data_extend(dataframe, extend_num):
    if extend_num == 0:
        return dataframe
    else:
        data_extend = pd.DataFrame()
        for i in range(extend_num - 1):
            data_extend = pd.concat((data_extend, dataframe), axis=0, ignore_index=True)
        return pd.concat((dataframe, data_extend), axis=0, ignore_index=True)
