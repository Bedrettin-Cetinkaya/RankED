import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss

def _expand_onehot_labels(labels, label_weights, label_channels):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

'''
def dice_Loss(inputs, targets, cuda=True, balance=1):
    n, c, h, w = inputs.size()
    smooth=1
    #inputs = torch.sigmoid(inputs)  # F.sigmoid(inputs)

    input_flat=inputs.view(-1)
    target_flat=targets.view(-1)

    intersecion=input_flat*target_flat
    unionsection=input_flat.pow(2).sum()+target_flat.pow(2).sum()+smooth
    loss=unionsection/(2*intersecion.sum()+smooth)
    loss=loss.sum()

    return loss
'''
'''
def dice_Loss(predict, target, cuda=True, balance=1):
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    
    num = torch.sum(torch.mul(predict, target))*2 + 1
    den = torch.sum(predict.pow(2) + target.pow(2)) + 1
    dice = num / den
    loss = 1 - dice

    return loss.sum()
'''
def dice_Loss(predict, target, cuda=True, balance=1):
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    
    num = torch.sum(torch.mul(predict, target))*2
    den = torch.sum(predict.pow(2) + target.pow(2))
    dice = den / num 

    return dice.sum()
    
def hed_loss_(pred,
             label,
             weight=None,
             reduction='mean',
             avg_factor=None,
             class_weight=None):
    """Calculate the binary CrossEntropy loss with weights.
    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    if weight is not None:
        weight = weight.float()

    total_loss = 0
    label = label.unsqueeze(1)
    batch, channel_num, imh, imw = pred.shape

    for b_i in range(batch):
        p = pred[b_i, :, :, :].unsqueeze(1)
        t = label[b_i, :, :, :].unsqueeze(1)
        mask = (t > 0.5).float()
        b, c, h, w = mask.shape
        #print(mask.size())
        #print("--")
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        class_weight = torch.zeros_like(mask)
        #print(mask)
        #print(num_pos)
        #print(num_neg)
        #print(mask.size())
        #print("----")
        class_weight[t > 0.5] = num_neg / (num_pos + num_neg)
        class_weight[t <= 0.5] = num_pos / (num_pos + num_neg)
        # weighted element-wise losses
        loss = F.binary_cross_entropy(p, t.float(), weight=class_weight, reduction='none')
        # do the reduction for the weighted loss
        #loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
        loss = torch.sum(loss)
        total_loss = total_loss + loss

    return total_loss

def hed_loss(pred,
             label,
             weight=None,
             reduction='mean',
             avg_factor=None,
             class_weight=None):
    """Calculate the binary CrossEntropy loss with weights.
    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    if weight is not None:
        weight = weight.float()

    total_loss = 0
    label = label.unsqueeze(1)
    batch, channel_num, imh, imw = pred.shape

    #print(label.size())
    #print(pred.size())
    for b_i in range(batch):
        p = pred[b_i, :, :, :].unsqueeze(1)
        t = label[b_i, :, :, :].unsqueeze(1)
        #print(p.size())
        #print(t.size())
        mask = (t > 0.5).float()
        b, c, h, w = mask.shape
        #print(mask)
        #print(mask.size())
        #print("--")
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        class_weight = torch.zeros_like(mask)
        #print(mask)
        #print(num_pos)
        #print(num_neg)
        #print(mask.size())
        #print("----")
        class_weight[t > 0.5] = num_neg / (num_pos + num_neg)
        class_weight[t <= 0.5] = num_pos / (num_pos + num_neg)
        #print((p>1).any())
        #print((p<0).any())
        # weighted element-wise losses
        loss = F.binary_cross_entropy(p, t.float(), weight=class_weight, reduction='none')
        #print("adssad")
        # do the reduction for the weighted loss
        #loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
        loss = torch.sum(loss)
        total_loss = total_loss + loss
        #print("---")
    #print("***")
    return total_loss


def balanced_mse(pred,
             label,
             weight=None,
             reduction='mean',
             avg_factor=None,
             class_weight=None):
             
    if weight is not None:
        weight = weight.float()

    total_loss = 0
    label = label.unsqueeze(1)
    batch, channel_num, imh, imw = pred.shape

    #print(label.size())
    #print(pred.size())
    for b_i in range(batch):
        p = pred[b_i, :, :, :].unsqueeze(1)
        t = label[b_i, :, :, :].unsqueeze(1)
        #print(p.size())
        #print(t.size())
        mask = (t > 0).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        class_weight = torch.zeros_like(mask)
        class_weight[t > 0.] = num_neg / (num_pos + num_neg)
        class_weight[t <= 0.] = num_pos / (num_pos + num_neg)
        # weighted element-wise losses
        loss = torch.sum(weight * ((pred - label) ** 2))
        total_loss = total_loss + loss
        #print("---")
    #print("***")
    return total_loss


class BMSE(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(BMSE, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        self.cls_criterion = balanced_mse
        self.f1_criterion = dice_Loss

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:

            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss_cls = self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor)
        #loss_f1 = self.f1_criterion(cls_score,label)

        #print(loss_cls)
        #print(loss_f1)
        #print("---", flush=True)
        #loss_cls = self.loss_weight * ( 0.0001 * loss_cls + loss_f1)
        #loss_cls = loss_f1 * 0.25
        return loss_cls
