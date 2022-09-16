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

        dataloader_train = dataloader_creator(dataset_train, label_train, train_index)
        dataloader_val = dataloader_creator(dataset_train, label_train, val_index)

        model.apply(reset_weights)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(path=f'saved_models_pilot/{name}.pth', patience=2500, verbose=True)

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
    dataloader_test = dataloader_creator(dataset_test, label_test, train=False)
    model = torch.load(f"saved_models_pilot/{name}.pth")
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
    model = torch.load(f"saved_models_pilot/{name}.pth")
    dataset_test = torch.from_numpy(dataset_test).to(device).float()

    explainer_shap = shap.DeepExplainer(model, dataset_test)
    shap_values = explainer_shap.shap_values(dataset_test, ranked_outputs=None)

    shap.summary_plot(shap_values, plot_type='bar')
    plt.savefig(f"./shap_plots_pilot/{name}.png")  # save fig for every models


def metaclassifier(device, names, dataset_train, label_train, dataset_test, label_test, epochs, val):
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


        # Model_32_64_128_32 = torch.load("saved_models/32-64-128-32.pth").to(device).eval()
        # Model_64_32_32_16_8 = torch.load("saved_models/64-32-32-16-8.pth").to(device).eval()
        # Model_64_64_32_16_8 = torch.load("saved_models/64-64-32-16-8.pth").to(device).eval()
        # Model_128_32_16_16_8 = torch.load("saved_models/128-32-16-16-8.pth").to(device).eval()
        # Model_128_32_32_16_8 = torch.load("saved_models/128-32-32-16-8.pth").to(device).eval()
        # Model_128_64_32 = torch.load("saved_models/128-64-32.pth").to(device).eval()
        # Model_128_64_32_16 = torch.load("saved_models/128-64-32-16.pth").to(device).eval()
        # Model_128_64_32_16_8 = torch.load("saved_models/128-64-32-16-8.pth").to(device).eval()
        # Model_128_64_32_32_8 = torch.load("saved_models/128-64-32-32-8.pth").to(device).eval()
        # Model_128_64_32_32_16 = torch.load("saved_models/128-64-32-32-16.pth").to(device).eval()
        # Model_128_64_64_16_8 = torch.load("saved_models/128-64-64-16-8.pth").to(device).eval()
        # Model_128_64_64_32_8 = torch.load("saved_models/128-64-64-32-8.pth").to(device).eval()
        # Model_128_64_64_32_16 = torch.load("saved_models/128-64-64-32-16.pth").to(device).eval()
        # Model_128_128_32_16_8 = torch.load("saved_models/128-128-32-16-8.pth").to(device).eval()
        # Model_128_128_64 = torch.load("saved_models/128-128-64.pth").to(device).eval()
        # Model_128_128_64_16_8 = torch.load("saved_models/128-128-64-16-8.pth").to(device).eval()
        # Model_128_128_64_32 = torch.load("saved_models/128-128-64-32.pth").to(device).eval()
        # Model_128_128_64_32_8 = torch.load("saved_models/128-128-64-32-8.pth").to(device).eval()
        # Model_128_128_64_32_16 = torch.load("saved_models/128-128-64-32-16.pth").to(device).eval()
        # Model_128_256_128 = torch.load("saved_models/128-256-128.pth").to(device).eval()
        # Model_128_256_128_64 = torch.load("saved_models/128-256-128-64.pth").to(device).eval()
        # Model_128_256_128_64_16 = torch.load("saved_models/128-256-128-64-16.pth").to(device).eval()

        models = []
        for name in names:
            models.append(torch.load(f"saved_models_pilot/{name}.pth").to(device).eval())

        # meta = Meta().to(device)
        # meta.apply(reset_weights)
        #
        # loss_fn = nn.BCELoss()
        # optimizer = torch.optim.Adam(meta.parameters(), lr=0.001)
        # early_stopping = EarlyStopping(path='saved_models_pilot/meta.pth', patience=200, verbose=True)
        #
        # for epoch in range(epochs):
        #     meta.train()
        #     for x, y in dataloader_train:
        #         x = x.to(device)
        #         y = y.to(device)
        #         optimizer.zero_grad()
        #
        #         # output_1 = Model_32_64_128_32(x)
        #         # output_2 = Model_64_32_32_16_8(x)
        #         # output_3 = Model_64_64_32_16_8(x)
        #         # output_4 = Model_128_32_16_16_8(x)
        #         # output_5 = Model_128_32_32_16_8(x)
        #         # output_6 = Model_128_64_32(x)
        #         # output_7 = Model_128_64_32_16(x)
        #         # output_8 = Model_128_64_32_16_8(x)
        #         # output_9 = Model_128_64_32_32_8(x)
        #         # output_10 = Model_128_64_32_32_16(x)
        #         # output_11 = Model_128_64_64_16_8(x)
        #         # output_12 = Model_128_64_64_32_8(x)
        #         # output_13 = Model_128_64_64_32_16(x)
        #         # output_14 = Model_128_128_32_16_8(x)
        #         # output_15 = Model_128_128_64(x)
        #         # output_16 = Model_128_128_64_16_8(x)
        #         # output_17 = Model_128_128_64_32(x)
        #         # output_18 = Model_128_128_64_32_8(x)
        #         # output_19 = Model_128_128_64_32_16(x)
        #         # output_20 = Model_128_256_128(x)
        #         # output_21 = Model_128_256_128_64(x)
        #         # output_22 = Model_128_256_128_64_16(x)
        #
        #         outputs = []
        #         for model in models:
        #             outputs.append(model(x))
        #
        #         # meta_input = torch.tensor(
        #         #     [output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8, output_9,
        #         #      output_10, output_11, output_12, output_13, output_14, output_15, output_16, output_17,
        #         #      output_18, output_19, output_20, output_21, output_22], device=device)
        #
        #         meta_input = torch.tensor(outputs, device=device)
        #
        #         pred = meta(meta_input)
        #         pred = pred.unsqueeze(1)
        #
        #         loss_1 = loss_fn(pred, y)
        #         loss_2 = contrastive_loss(pred, y)
        #         loss = loss_1 + loss_2
        #
        #         loss.backward()
        #         optimizer.step()
        #
        #     if val:
        #         meta.eval()
        #         with torch.no_grad():
        #             for x, y in dataloader_val:
        #                 x = x.to(device)
        #                 y = y.to(device)
        #
        #                 # output_1 = Model_32_64_128_32(x)
        #                 # output_2 = Model_64_32_32_16_8(x)
        #                 # output_3 = Model_64_64_32_16_8(x)
        #                 # output_4 = Model_128_32_16_16_8(x)
        #                 # output_5 = Model_128_32_32_16_8(x)
        #                 # output_6 = Model_128_64_32(x)
        #                 # output_7 = Model_128_64_32_16(x)
        #                 # output_8 = Model_128_64_32_16_8(x)
        #                 # output_9 = Model_128_64_32_32_8(x)
        #                 # output_10 = Model_128_64_32_32_16(x)
        #                 # output_11 = Model_128_64_64_16_8(x)
        #                 # output_12 = Model_128_64_64_32_8(x)
        #                 # output_13 = Model_128_64_64_32_16(x)
        #                 # output_14 = Model_128_128_32_16_8(x)
        #                 # output_15 = Model_128_128_64(x)
        #                 # output_16 = Model_128_128_64_16_8(x)
        #                 # output_17 = Model_128_128_64_32(x)
        #                 # output_18 = Model_128_128_64_32_8(x)
        #                 # output_19 = Model_128_128_64_32_16(x)
        #                 # output_20 = Model_128_256_128(x)
        #                 # output_21 = Model_128_256_128_64(x)
        #                 # output_22 = Model_128_256_128_64_16(x)
        #
        #                 outputs = []
        #                 for model in models:
        #                     outputs.append(model(x))
        #
        #                 # meta_input = torch.tensor(
        #                 #     [output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8, output_9,
        #                 #      output_10, output_11, output_12, output_13, output_14, output_15, output_16, output_17,
        #                 #      output_18, output_19, output_20, output_21, output_22], device=device)
        #
        #                 meta_input = torch.tensor(outputs, device=device)
        #
        #                 pred = meta(meta_input)
        #                 pred = pred.unsqueeze(1)
        #
        #                 loss_val_1 = loss_fn(pred, y)
        #                 loss_val_2 = contrastive_loss(pred, y)
        #                 loss_val = loss_val_1 + loss_val_2
        #
        #         print(f"Epoch:  {epoch + 1:4d}, loss: {loss:.6f}, loss_val: {loss_val:.6f}")
        #         wandb.log({
        #             "Training Loss": loss,
        #             "Validation Loss:": loss_val,
        #         })
        #     else:
        #         wandb.log({
        #             "Training Loss": loss,
        #         })
        #     early_stopping(loss_val, meta)
        #     if early_stopping.early_stop:
        #         print("Early Stopping!")
        #         break

    dataloader_test = dataloader_creator(dataset_test, label_test, train=False)

    dataset_float = torch.FloatTensor(dataset_test)
    label_float = torch.FloatTensor(label_test)
    label_float = label_float.unsqueeze(1)
    tensor = TensorDataset(dataset_float, label_float)
    dataloader_test = DataLoader(tensor, batch_size=1)

    model = torch.load("saved_models_pilot/meta.pth")


    with torch.no_grad():
        total, correct = 0, 0
        for x, y in dataloader_test:
            x = x.to(device)
            y = y.to(device)

            # output_1 = Model_32_64_128_32(x)
            # output_2 = Model_64_32_32_16_8(x)
            # output_3 = Model_64_64_32_16_8(x)
            # output_4 = Model_128_32_16_16_8(x)
            # output_5 = Model_128_32_32_16_8(x)
            # output_6 = Model_128_64_32(x)
            # output_7 = Model_128_64_32_16(x)
            # output_8 = Model_128_64_32_16_8(x)
            # output_9 = Model_128_64_32_32_8(x)
            # output_10 = Model_128_64_32_32_16(x)
            # output_11 = Model_128_64_64_16_8(x)
            # output_12 = Model_128_64_64_32_8(x)
            # output_13 = Model_128_64_64_32_16(x)
            # output_14 = Model_128_128_32_16_8(x)
            # output_15 = Model_128_128_64(x)
            # output_16 = Model_128_128_64_16_8(x)
            # output_17 = Model_128_128_64_32(x)
            # output_18 = Model_128_128_64_32_8(x)
            # output_19 = Model_128_128_64_32_16(x)
            # output_20 = Model_128_256_128(x)
            # output_21 = Model_128_256_128_64(x)
            # output_22 = Model_128_256_128_64_16(x)

            outputs = []
            for loaded_model in models:
                outputs.append(loaded_model(x))

            # meta_input = torch.tensor(
            #     [output_1, output_2, output_3, output_4, output_5, output_6, output_7, output_8, output_9,
            #      output_10, output_11, output_12, output_13, output_14, output_15, output_16, output_17,
            #      output_18, output_19, output_20, output_21, output_22], device=device)

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
