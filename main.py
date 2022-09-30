import os
import torch
import wandb

from dataloader import dataset_creator
from engine import train, evaluate_accuracy, evaluate_shap
from utils import create_models
from metaclassifier import metaclassifier, meta_evaluate_shap

from missing_values import missing_main

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("학습을 진행하는 기기: ", device)

    # wandb.init(project='SSNHL_pilot')

    # target = 'AAO-HNS guideline'
    # target = 'AAO criteria'
    target = 'Siegel criteria'

    epochs = 10000
    meta_epochs = 200

    dataset_train, dataset_test, label_train, label_test = dataset_creator(target, excel='1st')
    models, names = create_models()

    # for index, model in enumerate(models):
    #     wandb.init(project='AAO-HNS guideline predicted', name=names[index])
    #     train(device, model, names[index], dataset_train, label_train, epochs, val=True)
    #     evaluate_accuracy(device, names[index], dataset_test, label_test)
    #     evaluate_shap(device, names[index], dataset_test, label_test)
    #     wandb.finish()

    wandb.init(project='Siegel criteria predicted logits', name='MetaClassifier')
    metaclassifier(device, names, dataset_train, label_train, dataset_test, label_test, meta_epochs, val=True)
    # meta_evaluate_shap(device, models, dataset_test, label_test)

    # missing_main(device, models, names)
