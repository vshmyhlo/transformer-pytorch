import torch.nn.functional as F


def loss(y_top, y, padding_idx, reduce=True):
  not_padding = y != padding_idx
  loss = F.cross_entropy(y_top[not_padding], y[not_padding], reduce=reduce)
  return loss


def accuracy(y_top, y, padding_idx, reduce=True):
  _, y_top = y_top.max(-1)
  not_padding = y != padding_idx
  eq = y_top[not_padding] == y[not_padding]

  if reduce:
    return eq.float().mean()
  else:
    return eq.float()
