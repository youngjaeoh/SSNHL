import numpy as np
import pandas as pd
import torch
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.autograd.set_detect_anomaly(True)


def dataset_cleaner(dataset, target):
    # dataset = dataset_.dropna(subset=[target])

    minmax = MinMaxScaler()
    dataset['나이'] = minmax.fit_transform(dataset['나이'].values.reshape(-1, 1))
    dataset['Onset of treatment'] = minmax.fit_transform(dataset['Onset of treatment'].values.reshape(-1, 1))
    dataset['청력검사 (환측) Initial 500Hz'] = minmax.fit_transform(dataset['청력검사 (환측) Initial 500Hz'].values.reshape(-1, 1))
    dataset['청력검사 (환측) Initial 1000Hz'] = minmax.fit_transform(
        dataset['청력검사 (환측) Initial 1000Hz'].values.reshape(-1, 1))
    dataset['청력검사 (환측) Initial2000Hz'] = minmax.fit_transform(dataset['청력검사 (환측) Initial2000Hz'].values.reshape(-1, 1))
    dataset['청력검사 (환측) Initial4000Hz'] = minmax.fit_transform(dataset['청력검사 (환측) Initial4000Hz'].values.reshape(-1, 1))
    dataset['청력검사 (환측)  6분법(검사결과에 적힌대로)'] = minmax.fit_transform(
        dataset['청력검사 (환측)  6분법(검사결과에 적힌대로)'].values.reshape(-1, 1))
    dataset['청력검사 (환측) Initial WRS (%)'] = minmax.fit_transform(
        dataset['청력검사 (환측) Initial WRS (%)'].values.reshape(-1, 1))
    dataset['청력검사 (건측) Initial 500Hz'] = minmax.fit_transform(dataset['청력검사 (건측) Initial 500Hz'].values.reshape(-1, 1))
    dataset['청력검사 (건측) Initial 1000Hz'] = minmax.fit_transform(
        dataset['청력검사 (건측) Initial 1000Hz'].values.reshape(-1, 1))
    dataset['청력검사 (건측) Initial2000Hz'] = minmax.fit_transform(dataset['청력검사 (건측) Initial2000Hz'].values.reshape(-1, 1))
    dataset['청력검사 (건측) Initial4000Hz'] = minmax.fit_transform(dataset['청력검사 (건측) Initial4000Hz'].values.reshape(-1, 1))
    dataset['청력검사 (건측)  6분법(검사결과에 적힌대로)'] = minmax.fit_transform(
        dataset['청력검사 (건측)  6분법(검사결과에 적힌대로)'].values.reshape(-1, 1))
    dataset['청력검사 (건측) Initial WRS (%)'] = minmax.fit_transform(
        dataset['청력검사 (건측) Initial WRS (%)'].values.reshape(-1, 1))
    dataset['치료후 청력'] = minmax.fit_transform(dataset['치료후 청력'].values.reshape(-1, 1))
    dataset['청력검사 (환측) 치료후 500Hz'] = minmax.fit_transform(dataset['청력검사 (환측) 치료후 500Hz'].values.reshape(-1, 1))
    dataset['청력검사 (환측) 치료후 1000Hz'] = minmax.fit_transform(dataset['청력검사 (환측) 치료후 1000Hz'].values.reshape(-1, 1))
    dataset['청력검사 (환측) 치료후 2000Hz'] = minmax.fit_transform(dataset['청력검사 (환측) 치료후 2000Hz'].values.reshape(-1, 1))
    dataset['청력검사 (환측) 치료후 4000Hz'] = minmax.fit_transform(dataset['청력검사 (환측) 치료후 4000Hz'].values.reshape(-1, 1))
    dataset['청력검사 (환측) 치료후 6분법(검사결과에 적힌대로)'] = minmax.fit_transform(
        dataset['청력검사 (환측) 치료후 6분법(검사결과에 적힌대로)'].values.reshape(-1, 1))
    dataset['청력검사 (환측) 치료후 WRS (%)'] = minmax.fit_transform(dataset['청력검사 (환측) 치료후 WRS (%)'].values.reshape(-1, 1))

    del dataset['Number']
    del dataset['등록번호']
    del dataset['이름']
    del dataset['진단']
    del dataset['청력검사 (환측) Initial SRT']

    dataset = dataset.replace({'AAO criteria': 'A'}, 0)
    dataset = dataset.replace({'AAO criteria': 'B'}, 0)
    dataset = dataset.replace({'AAO criteria': 'C'}, 1)
    dataset = dataset.replace({'AAO criteria': 'D'}, 1)

    dataset_ = dataset.loc[dataset[target].notnull()]
    dataset_predict = dataset.loc[dataset[target].isna()]

    label = dataset_[target]
    del dataset_[target]
    del dataset_predict[target]

    print(dataset_.columns)
    print(dataset_.shape)
    print(dataset_predict.shape)

    return label, dataset_, dataset_predict


def dataset_creator(target):
    dataset = pd.read_excel('../SSNHL_data.xlsx', sheet_name='SSNHL_최종(결측값O)')

    label, dataset, dataset_predict = dataset_cleaner(dataset, target)
    label = label.values

    imputer = IterativeImputer(random_state=0)
    imputer.fit(dataset)
    dataset = imputer.transform(dataset)
    print('Missing: %d' % sum(np.isnan(dataset).flatten()))
    print(dataset.shape)

    dataset_train, dataset_test, label_train, label_test = train_test_split(dataset,
                                                                            label,
                                                                            test_size=0.2,
                                                                            random_state=1)
    return dataset_train, dataset_test, label_train, label_test, dataset_predict


def dataloader_creator(dataset, label, batch_size, index=0, train=True):
    if train:
        dataset = dataset[index]
        label = label[index]

    dataset_float = torch.FloatTensor(dataset)
    label_float = torch.FloatTensor(label)
    label_float = label_float.unsqueeze(1)
    tensor = TensorDataset(dataset_float, label_float)
    dl = DataLoader(tensor, batch_size=batch_size, shuffle=True, drop_last=False)

    return dl


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 48),
            nn.ReLU(),
            nn.Linear(48, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.network(x)
        return y


def missing_main(device, models, names):
    target = '심혈관질환'

    dataset_train, dataset_test, label_train, label_test, dataset_predict = dataset_creator(target)

    # wandb.init(project='SSNHL_MV', name=target)
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    for count, (train_index, val_index) in enumerate(kfold.split(dataset_train)):
        dataloader_train = dataloader_creator(dataset_train, label_train, batch_size=256, index=train_index)
        dataloader_val = dataloader_creator(dataset_train, label_train, batch_size=256, index=val_index)
