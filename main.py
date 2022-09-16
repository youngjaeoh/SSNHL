import os
import torch
import wandb

from dataloader import dataset_creator
from engine import train, evaluate_accuracy, evaluate_shap, metaclassifier
from utils import create_models

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("학습을 진행하는 기기: ", device)

    wandb.init(project='SSNHL_pilot')

    target = 'AAO-HNS guideline'

    epochs = 10
    meta_epochs = 5

    dataset_train, dataset_test, label_train, label_test = dataset_creator(target)
    models, names = create_models()

    # for index, model in enumerate(models):
    #     train(device, model, names[index], dataset_train, label_train, epochs, val=True)
    #     evaluate_accuracy(device, names[index], dataset_test, label_test)
    #     evaluate_shap(device, names[index], dataset_test, label_test)

    metaclassifier(device, names, dataset_train, label_train, dataset_test, label_test, meta_epochs, val=True)
