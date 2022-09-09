from engine import train, evaluate_accuracy, evaluate_shap
from model import Net
from dataloader import dataset_creator
import torch
import os
import wandb

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("학습을 진행하는 기기: ", device)

    wandb.init(project='SSNHL')

    target = 'AAO-HNS guideline'

    model = Net()
    epochs = 1000

    dataset_train, dataset_test, label_train, label_test = dataset_creator(target)
    train(device, model, dataset_train, label_train, epochs)
    evaluate_accuracy(device, dataset_test, label_test)
    evaluate_shap(device, dataset_test, label_test)
