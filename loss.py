import torch


epsilon = 1e-6
def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
    """
    Focal loss
    : param y_pred: input prediction
    : param y_true: input target
    : param alpha: balancing positive and negative samples, default=0.25
    : param gamma: penalizing wrong predictions, default=2
    """
    # alpha balance weight for unbalanced positive and negative samples
    # clip to prevent NaN's and Inf's
    y_pred_flatten = torch.clamp(y_pred, min=epsilon, max=1. - epsilon)
    y_pred_flatten = y_pred_flatten.view(-1).float()
    y_true_flatten = y_true.detach()
    y_true_flatten = y_true_flatten.view(-1).float()
    loss = 0

    idcs = (y_true_flatten > 0)
    y_true_pos = y_true_flatten[idcs]
    y_pred_pos = y_pred_flatten[idcs]
    y_true_neg = y_true_flatten[~idcs]
    y_pred_neg = y_pred_flatten[~idcs]

    if y_pred_pos.size(0) != 0 and y_true_pos.size(0) != 0:
        # positive samples
        logpt = torch.log(y_pred_pos)
        loss += -1. * \
            torch.mean(torch.pow((1. - y_pred_pos), gamma) * logpt) * alpha

    if y_pred_neg.size(0) != 0 and y_true_neg.size(0) != 0:
        # negative samples
        logpt2 = torch.log(1. - y_pred_neg)
        loss += -1. * torch.mean(torch.pow(y_pred_neg,
                                 gamma) * logpt2) * (1. - alpha)

    return loss