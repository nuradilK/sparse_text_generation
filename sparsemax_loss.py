import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.functional import relu

def _make_ix_like(X, dim):
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)

def _sparsemax_threshold_and_support(X, dim=-1, k=None):
    """Core computation for sparsemax: optimal threshold and support size.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.
    dim : int
        The dimension along which to apply sparsemax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    support_size : torch LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """

    if k is None or k >= X.shape[dim]:  # do full sort
        topk, _ = torch.sort(X, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(X, k=k, dim=dim)

    topk_cumsum = topk.cumsum(dim) - 1
    rhos = _make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(X.dtype)

    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)

        if torch.any(unsolved):
            in_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _sparsemax_threshold_and_support(in_, dim=-1, k=2 * k)
            _roll_last(tau, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_

    return tau, support_size


class _GenericLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction="elementwise_mean"):
        assert reduction in ["elementwise_mean", "sum", "none"]
        self.reduction = reduction
        self.ignore_index = ignore_index
        super(_GenericLoss, self).__init__()

    def forward(self, X, target):
        loss = self.loss(X, target)
        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "elementwise_mean":
            loss = loss.sum() / size
        return loss
    
class SparsemaxFunction(Function):
    @classmethod
    def forward(cls, ctx, X, dim=-1, k=None):
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as softmax
        tau, supp_size = _sparsemax_threshold_and_support(X, dim=dim, k=k)
        output = torch.clamp(X - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None

    
def sparsemax(X, dim=-1, k=None):
    """sparsemax: normalizing sparse transform (a la softmax).
    Solves the projection:
        min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor.
    dim : int
        The dimension along which to apply sparsemax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """

    return SparsemaxFunction.apply(X, dim, k)
    
    
class SparsemaxLossFunction(nn.Module):
    def __init__(self):
        super(SparsemaxLossFunction, self).__init__()
    
    def forward(self, X, target, proj_args):
        """
        X (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert X.shape[0] == target.shape[0]
        
        cumsum = X.sort(descending=True, dim=1)[0]
        cumsum = (cumsum ** 2).cumsum(dim=1)
        tau, supp = _sparsemax_threshold_and_support(X, dim=1)
        loss = -X[torch.arange(len(X)), target] + (1 + (cumsum[torch.arange(len(X)), supp - 1] - tau ** 2 * supp).sum(dim=1)) / 2

        return loss

class SparsemaxLoss(_GenericLoss):
    def __init__(self, k=None, ignore_index=-100, reduction="elementwise_mean"):
        self.k = k
#         self.SparsemaxLossFunc = SparsemaxLossFunction()
        super(SparsemaxLoss, self).__init__(ignore_index, reduction)

    def loss(self, X, target):
        return SparsemaxLossFunction()(X, target, self.k)