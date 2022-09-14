import torch
from torch import nn
from criterion import contrastive_loss
from sklearn.model_selection import KFold
from dataloader import dataloader_creator
from utils import EarlyStopping, reset_weights
import wandb
import shap
import matplotlib.pyplot as plt
import numpy as np
from model import Meta
from torch.utils.data import TensorDataset, DataLoader


def train(device, model, dataset_train, label_train, epochs, val):
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    model = model.to(device)

    for count, (train_index, val_index) in enumerate(kfold.split(dataset_train)):

        dataloader_train = dataloader_creator(dataset_train, label_train, train_index)
        dataloader_val = dataloader_creator(dataset_train, label_train, val_index)

        model.apply(reset_weights)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(path='saved_models/model.pth', patience=2500, verbose=True)

        for epoch in range(epochs):
            loss_val = train_one_epoch(device, model, dataloader_train, dataloader_val, epoch, loss_fn, optimizer,
                                       early_stopping, val)

            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early Stopping!")
                break


def train_one_epoch(device, model, dataloader_train, dataloader_val, epoch, loss_fn, optimizer, early_stopping, val):
    model.train()
    for x, y in dataloader_train:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss_1 = loss_fn(y_pred, y)
        loss_2 = contrastive_loss(y_pred, y)
        loss = loss_1 + loss_2

        loss.backward()
        optimizer.step()

    if val:
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

        wandb.log({
            "Training Loss": loss,
            "Validation Loss:": loss_val,
        })
    else:
        wandb.log({
            "Training Loss": loss,
        })

    return loss_val


def evaluate_accuracy(device, dataset_test, label_test):
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

    wandb.log({
        "Accuracy": accuracy
    })


def evaluate_shap(device, dataset_test, label_test):
    model = torch.load("saved_models/model.pth")
    dataset_test = torch.from_numpy(dataset_test).to(device).float()

    explainer_shap = shap.DeepExplainer(model, dataset_test)
    shap_values = explainer_shap.shap_values(dataset_test, ranked_outputs=None)

    shap.summary_plot(shap_values, plot_type='bar')
    # plt.savefig("./shap_plots/model_{}.png".format(index + 1))  # save fig for every models


def metaclassifier(device, dataset_train, label_train, dataset_test, label_test, epochs, val):
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)

    for count, (train_index, val_index) in enumerate(kfold.split(dataset_train)):

        # dataloader_train = dataloader_creator(dataset_train, label_train, train_index)

        dataset = dataset_train[train_index]
        label = label_train[train_index]
        dataset_float = torch.FloatTensor(dataset)
        label_float = torch.FloatTensor(label)
        label_float = label_float.unsqueeze(1)
        tensor = TensorDataset(dataset_float, label_float)
        dataloader_train = DataLoader(tensor, batch_size=1)

        # dataloader_val = dataloader_creator(dataset_train, label_train, val_index)

        dataset = dataset_train[val_index]
        label = label_train[val_index]
        dataset_float = torch.FloatTensor(dataset)
        label_float = torch.FloatTensor(label)
        label_float = label_float.unsqueeze(1)
        tensor = TensorDataset(dataset_float, label_float)
        dataloader_val = DataLoader(tensor, batch_size=1)

        Logistic_Regression = torch.load("saved_models/Logistic_Regression.pth")
        DL_1 = torch.load("saved_models/128-64-32-8.pth")

        Logistic_Regression.to(device)
        Logistic_Regression.eval()

        DL_1.to(device)
        DL_1.eval()

        meta = Meta().to(device)
        meta.apply(reset_weights)

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(meta.parameters(), lr=0.001)
        early_stopping = EarlyStopping(path='saved_models/meta.pth', patience=2500, verbose=True)

        for epoch in range(epochs):
            meta.train()
            for x, y in dataloader_train:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                output_1 = Logistic_Regression(x)
                output_2 = DL_1(x)

                meta_input = torch.FloatTensor([output_1, output_2])
                meta_input = meta_input.to(device)

                pred = meta(meta_input)
                pred = pred.unsqueeze(1)

                loss_1 = loss_fn(pred, y)
                loss_2 = contrastive_loss(pred, y)
                loss = loss_1 + loss_2

                loss.backward()
                optimizer.step()

            if val:
                meta.eval()
                with torch.no_grad():
                    for x, y in dataloader_val:
                        x = x.to(device)
                        y = y.to(device)
                        meta_input = list()
                        output_1 = Logistic_Regression(x)
                        output_2 = DL_1(x)
                        meta_input = torch.FloatTensor([output_1, output_2])
                        meta_input = meta_input.to(device)

                        pred = meta(meta_input)
                        pred = pred.unsqueeze(1)

                        loss_val_1 = loss_fn(pred, y)
                        loss_val_2 = contrastive_loss(pred, y)
                        loss_val = loss_val_1 + loss_val_2

                print(f"Epoch:  {epoch + 1:4d}, loss: {loss:.6f}, loss_val: {loss_val:.6f}")
                wandb.log({
                    "Training Loss": loss,
                    "Validation Loss:": loss_val,
                })
            else:
                wandb.log({
                    "Training Loss": loss,
                })
            early_stopping(loss_val, meta)
            if early_stopping.early_stop:
                print("Early Stopping!")
                break

    dataloader_test = dataloader_creator(dataset_test, label_test, train=False)

    dataset_float = torch.FloatTensor(dataset_test)
    label_float = torch.FloatTensor(label_test)
    label_float = label_float.unsqueeze(1)
    tensor = TensorDataset(dataset_float, label_float)
    dataloader_test = DataLoader(tensor, batch_size=1)

    model = torch.load("saved_models/meta.pth")
    model.to(device)
    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for x, y in dataloader_test:
            x = x.to(device)
            y = y.to(device)

            output_1 = Logistic_Regression(x)
            output_2 = DL_1(x)

            meta_input = torch.FloatTensor([output_1, output_2])
            meta_input = meta_input.to(device)

            output = model(meta_input)

            for index, i in enumerate(output):
                total += 1
                if i.item() > 0.5 and y[index] == 1:
                    correct += 1
                elif i.item() < 0.5 and y[index] == 0:
                    correct += 1
    accuracy = 100 * correct / total
    print('total: ', total)
    print('correct: ', correct)
    print(f"Accuracy: {accuracy:.6f}")

    wandb.log({
        "Meta_Accuracy": accuracy
    })
