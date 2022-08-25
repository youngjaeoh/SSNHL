import torch
from torch import nn
from criterion import contrastive_loss
from sklearn.model_selection import KFold
from dataloader import dataloader_creator

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, dataset_train, label_train):
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)

    for count, (train_index, val_index) in enumerate(kfold.split(dataset_train)):
        dataloader_train = dataloader_creator(dataset_train, label_train, train_index)
        dataloader_val = dataloader_creator(dataset_train, label_train, val_index)

        model.to(device)
        loss_fn = nn.BCELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(1000):
            train_one_epoch(model, dataloader_train, dataloader_val, epoch, loss_fn, optimizer)

        torch.save(model, './saved_models/{}.pth'.format('model'))


def train_one_epoch(model, dataloader_train, dataloader_val, epoch, loss_fn, optimizer):
    model.train()
    for x, y in dataloader_train:
        x.to(device)
        y.to(device)

        y_pred = model(x)
        loss_1 = loss_fn(y_pred, y)
        loss_2 = contrastive_loss(y_pred, y)
        loss = loss_1 + loss_2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for x, y in dataloader_val:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss_val_1 = loss_fn(output, y)
            loss_val_2 = contrastive_loss(output, y)
            loss_val = loss_val_1 + loss_val_2

    print(f"Epoch:  {epoch + 1:4d}, loss: {loss:.6f}, loss_val: {loss_val:.6f}")


def evaluate_accuracy(dataset_test, label_test):
    dataloader_test = dataloader_creator(dataset_test, label_test, train=False)
    model = torch.load("saved_models/model.pth")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in dataloader_test:
            x = x.to(device)
            y = y.to(device)
            output = model(x)

            total, correct = 0, 0
            for index, i in enumerate(output):
                total += 1
                if i.item() >= 0.5 and y[index] == 1:
                    correct += 1
                elif i.item() < 0.5 and y[index] == 0:
                    correct += 1
    accuracy = 100 * correct / total
    print('total: ', total)
    print('correct: ', correct)
    print(f"Accuracy: {accuracy:.6f}")