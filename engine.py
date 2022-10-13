import matplotlib.pyplot as plt
import shap
import torch
import wandb
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from criterion import contrastive_loss
from dataloader import dataloader_creator
from model import Meta
from utils import EarlyStopping, reset_weights


def train(device, model, name, dataset_train, label_train, epochs, val):
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    model = model.to(device)

    for count, (train_index, val_index) in enumerate(kfold.split(dataset_train)):

        dataloader_train = dataloader_creator(dataset_train, label_train, batch_size=256, index=train_index)
        dataloader_val = dataloader_creator(dataset_train, label_train, batch_size=256, index=val_index)

        model.apply(reset_weights)
        loss_fn = nn.BCELoss()
        # loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(path=f'outputs/1st/saved_models_HNS/{name}.pth', patience=2500, verbose=True)

        for epoch in range(epochs):
            loss_val = train_one_epoch(device, model, dataloader_train, dataloader_val, epoch, loss_fn, optimizer, val)

            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early Stopping!")
                break


def train_one_epoch(device, model, dataloader_train, dataloader_val, epoch, loss_fn, optimizer, val):
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

        wandb.log({
            "Training Loss": loss,
        })

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
            "Validation Loss:": loss_val,
        })

    return loss_val


def evaluate_accuracy(device, name, dataset_test, label_test):
    dataloader_test = dataloader_creator(dataset_test, label_test, batch_size=256, train=False)
    model = torch.load(f"outputs/1st/saved_models_HNS/{name}.pth")
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


def evaluate_shap(device, name, dataset_test, label_test):
    model = torch.load(f"/home/youngjaeoh/nas/SSNHL/outputs/1st/saved_models_siegel/{name}.pth")
    dataset_test = torch.from_numpy(dataset_test).to(device).float()

    explainer_shap = shap.DeepExplainer(model, dataset_test)
    shap_values = explainer_shap.shap_values(dataset_test, ranked_outputs=None)

    shap.summary_plot(shap_values, max_display=100, plot_type='bar', show=False)
    plt.savefig(f"/home/youngjaeoh/nas/SSNHL/outputs/1st/shap_plots_siegel/{name}.png")  # save fig for every models



