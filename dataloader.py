import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def dataset_cleaner(dataset, target):
    if target == 'AAO criteria':
        del dataset['Siegel criteria']
        del dataset['AAO-HNS guideline']
    elif target == 'Siegel criteria':
        del dataset['AAO criteria']
        del dataset['AAO-HNS guideline']
    elif target == 'AAO-HNS guideline':
        del dataset['AAO criteria']
        del dataset['Siegel criteria']

    dataset = dataset.replace({target: 1}, 0)
    dataset = dataset.replace({target: 2}, 0)
    dataset = dataset.replace({target: 3}, 1)

    label = dataset[target]
    del dataset[target]
    del dataset['Number']
    del dataset['등록번호']
    del dataset['이름']
    del dataset['진단']
    del dataset['청력검사 (환측) Initial SRT']  # 추가로 삭제해봄

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
    # dataset['청력검사 (환측) Initial SRT'] = minmax.fit_transform(dataset['청력검사 (환측) Initial SRT'].values.reshape(-1, 1))
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
    #
    # print(dataset.columns)
    # print(dataset.shape)
    return label, dataset


def dataloader_creator(dataset, label, index=0, train=True):
    if train:
        dataset = dataset[index]
        label = label[index]

    dataset_float = torch.FloatTensor(dataset)
    label_float = torch.FloatTensor(label)
    label_float = label_float.unsqueeze(1)
    tensor = TensorDataset(dataset_float, label_float)
    dl = DataLoader(tensor, batch_size=256, shuffle=True, drop_last=False)

    return dl


def dataset_creator(target):
    dataset = pd.read_excel('../SSNHL_data.xlsx', sheet_name='SSNHL_최종(결측값제거)')
    label, dataset = dataset_cleaner(dataset, target)
    label = label.values
    dataset = dataset.values
    dataset_train, dataset_test, label_train, label_test = train_test_split(dataset,
                                                                            label,
                                                                            test_size=0.2,
                                                                            random_state=1)
    return dataset_train, dataset_test, label_train, label_test
