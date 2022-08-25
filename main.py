from engine import train, evaluate_accuracy
from model import Net
from dataloader import dataset_creator

if __name__ == "__main__":
    target = 'AAO-HNS guideline'
    # target = 'Siegel criteria'
    # target = 'AAO criteria'

    model = Net()
    # dataloader_train, dataloader_test = dataloader(target)
    # dataloader_train, dataloader_test = dataloader(target)

    dataset_train, dataset_test, label_train, label_test = dataset_creator(target)

    train(model, dataset_train, label_train)
    evaluate_accuracy(dataset_test, label_test)
