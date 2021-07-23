#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Union

import torch
import numpy as np

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['f1_per_frame', 'error_rate_per_frame']


_eps: float = torch.finfo(torch.float32).eps

def balanced_accuracy(y_hat: torch.Tensor, y:torch.Tensor) \
        -> torch.Tensor:

    """Uses 3d (B x #Sequences x Classes) indices of ground truth pos and neg
    as keys for a hash map via "raveling".
    Prediction accuracies are "thresholded" and become 0/1. Predictions indices
    are looked up in hash map to check for true positive, true negative, false postive
    false negative. Should scale better than N^2
    """

    threshhold = 0.55
    # o = torch.ones_like(y)
    # z = torch.zeros_like(y)

    pos = torch.nonzero(y).cpu() # B x S x nclass
    neg = torch.nonzero(torch.where(y == 1, 0, 1)).cpu()

    cols = [pos[:, c].numpy() for c in range(pos.shape[-1])]
    pos_ravel = np.ravel_multi_index(tuple(cols), dims=tuple(y.size()))
    pos_dict = dict.fromkeys(pos_ravel)

    cols = [neg[:, c].numpy() for c in range(neg.shape[-1])]
    neg_ravel = np.ravel_multi_index(tuple(cols), dims=tuple(y.size()))
    neg_dict = dict.fromkeys(neg_ravel)

    pred = torch.where(y_hat > threshhold, 1, 0).cpu()
    pred_pos = torch.nonzero(pred).cpu()
    cols = [pred_pos[:, c].numpy() for c in range(pred_pos.shape[-1])]
    pred_pos_ravel = np.ravel_multi_index(tuple(cols), dims=tuple(y.size()))

    pred_neg = torch.nonzero(torch.where(pred == 1, 0, 1)).cpu()
    cols = [pred_neg[:, c].numpy() for c in range(pred_neg.shape[-1])]
    pred_neg_ravel = np.ravel_multi_index(tuple(cols), dims=tuple(y.size()))

    true_pos = 0
    false_pos = 0
    for ind in pred_pos_ravel:
        if ind in pos_dict.keys():
            true_pos+=1
        else:
            false_pos+=1


    true_neg = 0
    false_neg = 0
    for ind in pred_neg_ravel:
        if ind in neg_dict.keys():
            true_neg+=1
        else:
            false_neg+=1


    sensitivity = true_pos / (true_pos+false_neg+1e-5)
    specificity = true_neg / (false_pos+true_neg+1e-5)

    bac = .5 * (sensitivity + specificity)

    # print(true_pos, true_neg, false_pos, false_neg)
    return bac


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) \
        -> torch.Tensor:

    return torch.eq(y, y_hat.round()).sum() / torch.numel(y)





def f1_per_frame(y_hat: torch.Tensor,
                 y_true: torch.Tensor) \
        -> torch.Tensor:
    """Gets the average per frame F1 score, based on\
    TP, FP, and FN, calculated from the `y_hat`\
    predictions and `y_true` ground truth values.

    :param y_hat: Predictions
    :type y_hat: torch.Tensor
    :param y_true: Ground truth values
    :type y_true: torch.Tensor
    :return: F1 score per frame
    :rtype: torch.Tensor
    """
    tp, _, fp, fn = _tp_tf_fp_fn(
        y_hat=y_hat, y_true=y_true,
        dim_sum=None)

    tp = tp.sum()
    fp = fp.sum()
    fn = fn.sum()
    the_f1 = _f1(tp=tp, fp=fp, fn=fn)

    return the_f1


def error_rate_per_frame(y_hat: torch.Tensor,
                         y_true: torch.Tensor) \
        -> torch.Tensor:
    """Calculates the error rate based on FN and FP,
    for one second.

    :param y_hat: Predictions.
    :type y_hat: torch.Tensor
    :param y_true: Ground truth.
    :type y_true: torch.Tensor
    :return: Error rate.
    :rtype: torch.Tensor
    """
    _, __, fp, fn = _tp_tf_fp_fn(
        y_hat=y_hat, y_true=y_true,
        dim_sum=-1)

    s = fn.min(fp).sum()
    d = fn.sub(fp).clamp_min(0).sum()
    i = fp.sub(fn).clamp_min(0).sum()
    n = y_true.sum() + _eps

    return (s + d + i)/n


def _f1(tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor) \
        -> torch.Tensor:
    """Gets the F1 score from the TP, FP, and FN.

    :param tp: TP
    :type tp: torch.Tensor
    :param fp: FP
    :type fp: torch.Tensor
    :param fn: FN
    :type fn: torch.Tensor
    :return: F1 score
    :rtype: torch.Tensor
    """
    if all([m.sum().item() == 0 for m in [tp, fp, fn]]):
        return torch.zeros(1).to(tp.device)

    f1_nominator = tp.mul(2)
    f1_denominator = tp.mul(2).add(fn).add(fp)

    return f1_nominator.div(f1_denominator + _eps)


def _tp_tf_fp_fn(y_hat: torch.Tensor,
                 y_true: torch.Tensor,
                 dim_sum: Union[int, None]) \
        -> Tuple[torch.Tensor, torch.Tensor,
                 torch.Tensor, torch.Tensor]:
    """Gets the true positive (TP), true negative (TN),\
    false positive (FP), and false negative (FN).

    :param y_hat: Predictions
    :type y_hat: torch.Tensor
    :param y_true: Ground truth values
    :type y_true: torch.Tensor
    :param dim_sum: Dimension to sum TP, TN, FP, and FN. If\
                    it is None, then the default behaviour from\
                    PyTorch`s sum is assumed.
    :type dim_sum: int|None
    :return: TP, TN, FP, FN.
    :rtype: (torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor)
    """
    y_hat_positive = y_hat.ge(0.5)
    y_hat_negative = y_hat.lt(0.5)

    y_true_positive = y_true.eq(1.)
    y_true_negative = y_true.eq(0.)

    tp = y_hat_positive.mul(y_true_positive).view(
        -1, y_hat_positive.size()[-1]
    ).float()

    tn = y_hat_negative.mul(y_true_negative).view(
        -1, y_hat_positive.size()[-1]
    ).float()

    fp = y_hat_positive.mul(y_true_negative).view(
        -1, y_hat_positive.size()[-1]
    ).float()

    fn = y_hat_negative.mul(y_true_positive).view(
        -1, y_hat_positive.size()[-1]
    ).float()

    if dim_sum is not None:
        tp = tp.sum(dim=dim_sum)
        tn = tn.sum(dim=dim_sum)
        fp = fp.sum(dim=dim_sum)
        fn = fn.sum(dim=dim_sum)

    return tp, tn, fp, fn

# EOF
