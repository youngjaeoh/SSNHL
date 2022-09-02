from engine import train, evaluate_accuracy
from model import Net
from dataloader import dataset_creator
import torch
import numpy as np
import random


if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)

    target = 'AAO-HNS guideline'

    model = Net()
    dataset_train, dataset_test, label_train, label_test = dataset_creator(target)

    train(model, dataset_train, label_train)
    evaluate_accuracy(dataset_test, label_test)
