import torch


def contrastive_loss(y_pred, y):
    square_pred = torch.square(y_pred)
    margin_square = torch.square(torch.maximum(1 - y_pred, torch.tensor(0)))
    loss_contrastive = torch.mean(y * square_pred + (1 - y) * margin_square)
    return loss_contrastive
