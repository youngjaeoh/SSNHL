import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import wandb
from sklearn.model_selection import KFold
from torch import nn

from criterion import contrastive_loss
from dataloader import dataloader_creator
from model import Meta
from utils import EarlyStopping, reset_weights


def meta_train(device, meta, models, dataset_train, label_train, epochs, val):
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    for count, (train_index, val_index) in enumerate(kfold.split(dataset_train)):

        dataloader_train = dataloader_creator(dataset_train, label_train, batch_size=1, index=train_index)
        dataloader_val = dataloader_creator(dataset_train, label_train, batch_size=1, index=val_index)

        meta.apply(reset_weights)

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(meta.parameters(), lr=0.001)
        early_stopping = EarlyStopping(path='outputs/1st/saved_models_siegel_logits/meta.pth', patience=20, verbose=True)

        for epoch in range(epochs):
            loss_val = meta_train_one_epoch(device, meta, models, dataloader_train, dataloader_val, epoch, loss_fn,
                                            optimizer, val)

            early_stopping(loss_val, meta)
            if early_stopping.early_stop:
                print("Early Stopping!")
                break


def meta_train_one_epoch(device, meta, models, dataloader_train, dataloader_val, epoch, loss_fn, optimizer, val):
    meta.train()
    for x, y in dataloader_train:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        outputs = []
        for model in models:
            outputs.append(model(x))

        meta_input = torch.tensor(outputs, device=device)

        pred = meta(meta_input)
        pred = pred.unsqueeze(1)

        loss_1 = loss_fn(pred, y)
        loss_2 = contrastive_loss(pred, y)
        loss = loss_1 + loss_2

        loss.backward()
        optimizer.step()

        wandb.log({
            "Training Loss": loss,
        })

    if val:
        meta.eval()
        with torch.no_grad():
            for x, y in dataloader_val:
                x = x.to(device)
                y = y.to(device)

                outputs = []
                for model in models:
                    outputs.append(model(x))

                meta_input = torch.tensor(outputs, device=device)

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

    return loss_val


def meta_evaluate_accuracy(device, models, dataset_test, label_test):
    dataloader_test = dataloader_creator(dataset_test, label_test, batch_size=1, train=False)
    model = torch.load("outputs/1st/saved_models_siegel_logits/meta.pth")
    with torch.no_grad():
        total, correct = 0, 0
        for x, y in dataloader_test:
            x = x.to(device)
            y = y.to(device)

            outputs = []
            for loaded_model in models:
                outputs.append(loaded_model(x))

            meta_input = torch.tensor(outputs, device=device)

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
        "Accuracy": accuracy
    })


def meta_evaluate_shap(device, models, dataset_test, label_test):
    meta_classifier = torch.load("outputs/1st/saved_models_siegel_logits/meta.pth")
    # dataset_test = torch.from_numpy(dataset_test).to(device).float()
    dataloader_test = dataloader_creator(dataset_test, label_test, batch_size=1, train=False)

    for x, y in dataloader_test:
        x = x.to(device)
        y = y.to(device)

        outputs = []
        for model in models:
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                outputs.append(model(x))

    # meta_input = torch.tensor(outputs, device=device)
    # meta_input = meta_input.unsqueeze(0)
    outputs = np.array(outputs)
    meta_input = torch.from_numpy(outputs).to(device).float()

    explainer_shap = shap.DeepExplainer(meta_classifier, meta_input)
    shap_values = explainer_shap.shap_values(meta_input, ranked_outputs=None)

    shap.summary_plot(shap_values, plot_type='bar')
    plt.savefig("./outputs/1st/shap_plots_siegel_logits/meta.png")


def metaclassifier(device, names, dataset_train, label_train, dataset_test, label_test, epochs, val):
    meta = Meta().to(device)
    models = []
    for name in names:
        models.append(torch.load(f"outputs/1st/saved_models_siegel_logits/{name}.pth").to(device).eval())

    meta_train(device, meta, models, dataset_train, label_train, epochs, val)
    meta_evaluate_accuracy(device, models, dataset_test, label_test)
