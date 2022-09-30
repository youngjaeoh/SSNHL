import numpy as np
import torch

import model


def create_models():
    Model_32_64_128_32 = model.Model_32_64_128_32()
    Model_64_32_32_16_8 = model.Model_64_32_32_16_8()
    Model_64_64_32_16_8 = model.Model_64_64_32_16_8()
    Model_128_32_16_16_8 = model.Model_128_32_16_16_8()
    Model_128_32_32_16_8 = model.Model_128_32_32_16_8()
    Model_128_64_32 = model.Model_128_64_32()
    Model_128_64_32_16 = model.Model_128_64_32_16()
    Model_128_64_32_16_8 = model.Model_128_64_32_16_8()
    Model_128_64_32_32_8 = model.Model_128_64_32_32_8()
    Model_128_64_32_32_16 = model.Model_128_64_32_32_16()
    Model_128_64_64_16_8 = model.Model_128_64_64_16_8()
    Model_128_64_64_32_8 = model.Model_128_64_64_32_8()
    Model_128_64_64_32_16 = model.Model_128_64_64_32_16()
    Model_128_128_32_16_8 = model.Model_128_128_32_16_8()
    Model_128_128_64 = model.Model_128_128_64()
    Model_128_128_64_16_8 = model.Model_128_128_64_16_8()
    Model_128_128_64_32 = model.Model_128_128_64_32()
    Model_128_128_64_32_8 = model.Model_128_128_64_32_8()
    Model_128_128_64_32_16 = model.Model_128_128_64_32_16()
    Model_128_256_128 = model.Model_128_256_128()
    Model_128_256_128_64 = model.Model_128_256_128_64()
    Model_128_256_128_64_16 = model.Model_128_256_128_64_16()

    models = [
        Model_32_64_128_32,
        Model_64_32_32_16_8,
        Model_64_64_32_16_8,
        Model_128_32_16_16_8,
        Model_128_32_32_16_8,
        Model_128_64_32,
        Model_128_64_32_16,
        Model_128_64_32_16_8,
        Model_128_64_32_32_8,
        Model_128_64_32_32_16,
        Model_128_64_64_16_8,
        Model_128_64_64_32_8,
        Model_128_64_64_32_16,
        Model_128_128_32_16_8,
        Model_128_128_64,
        Model_128_128_64_16_8,
        Model_128_128_64_32,
        Model_128_128_64_32_8,
        Model_128_128_64_32_16,
        Model_128_256_128,
        Model_128_256_128_64,
        Model_128_256_128_64_16]

    names = [
        "32-64-128-32",
        "64-32-32-16-8",
        "64-64-32-16-8",
        "128-32-16-16-8",
        "128-32-32-16-8",
        "128-64-32",
        "128-64-32-16",
        "128-64-32-16-8",
        "128-64-32-32-8",
        "128-64-32-32-16",
        "128-64-64-16-8",
        "128-64-64-32-8",
        "128-64-64-32-16",
        "128-128-32-16-8",
        "128-128-64",
        "128-128-64-16-8",
        "128-128-64-32",
        "128-128-64-32-8",
        "128-128-64-32-16",
        "128-256-128",
        "128-256-128-64",
        "128-256-128-64-16"]

    return models, names


class EarlyStopping:
    def __init__(self, path, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)  # 모델 전체 저장
        self.val_loss_min = val_loss


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
