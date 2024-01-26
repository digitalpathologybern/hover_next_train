import torch
from torch import nn
from typing import Optional

# from https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.

    Shape:
        - Pred: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is an integer
          representing correct classification :math:`target[i] \in [0, C)`.

    Example:
        >>> C = 5  # num_classes
        >>> pred = torch.randn(1, C, 3, 5, requires_grad=True)
        >>> target = torch.randint(C, (1, 3, 5))
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> output = criterion(pred, target)
        >>> output.backward()
    """

    def __init__(
        self,
        alpha: Optional[float],
        gamma: float = 2.0,
        reduction: str = "none",
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.alpha: Optional[float] = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.weight: Optional[torch.Tensor] = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(
            pred, target, self.alpha, self.gamma, self.reduction, self.weight
        )


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: Optional[float],
    gamma: float = 2.0,
    reduction: str = "none",
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        pred: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is an integer
          representing correct classification :math:`target[i] \in [0, C)`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.

    Return:
        the computed loss.

    Example:
        >>> C = 5  # num_classes
        >>> pred = torch.randn(1, C, 3, 5, requires_grad=True)
        >>> target = torch.randint(C, (1, 3, 5))
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> output = focal_loss(pred, target, **kwargs)
        >>> output.backward()
    """

    out_size = (pred.shape[0],) + pred.shape[2:]

    # create the labels one hot tensor
    target_one_hot = one_hot(
        target, num_classes=pred.shape[1], device=pred.device, dtype=pred.dtype
    )

    # compute softmax over the classes axis
    log_pred_soft = pred.log_softmax(1)

    # compute the actual focal loss
    loss_tmp = (
        -torch.pow(1.0 - log_pred_soft.exp(), gamma) * log_pred_soft * target_one_hot
    )

    num_of_classes = pred.shape[1]
    broadcast_dims = [-1] + [1] * len(pred.shape[2:])
    if alpha is not None:
        alpha_fac = torch.tensor(
            [1 - alpha] + [alpha] * (num_of_classes - 1),
            dtype=loss_tmp.dtype,
            device=loss_tmp.device,
        )
        alpha_fac = alpha_fac.view(broadcast_dims)
        loss_tmp = alpha_fac * loss_tmp

    if weight is not None:
        weight = weight.view(broadcast_dims)
        loss_tmp = weight * loss_tmp

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Input labels type is not a Tensor. Got {type(labels)}")

    if not labels.dtype == torch.int64:
        raise ValueError(
            f"labels must be of the same dtype torch.int64. Got: {labels.dtype}"
        )

    if num_classes < 1:
        raise ValueError(
            f"The number of classes must be bigger than one. Got: {num_classes}"
        )

    shape = labels.shape
    one_hot = torch.zeros(
        (shape[0], num_classes) + shape[1:], device=device, dtype=dtype
    )

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
