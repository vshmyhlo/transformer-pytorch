import torch.nn.functional as F


class Summary(object):
  def __init__(self, metrics):
    self.metrics = list(metrics)
    self.n_samples = 0

  def add(self, metrics):
    assert len(metrics) > 0
    assert len(metrics) == len(self.metrics)
    size = metrics[0].size()
    assert len(size) == 1
    self.n_samples += size[0]

    for i in range(len(metrics)):
      assert metrics[i].size() == size

      self.metrics[i] += metrics[i].sum()

  def calculate(self):
    if self.n_samples > 0:
      return [x / self.n_samples for x in self.metrics]
    else:
      return self.metrics


def loss(y_top, y, padding_idx):
  not_padding = y != padding_idx
  # TODO: ignore_index argument
  loss = F.cross_entropy(y_top[not_padding], y[not_padding], reduce=False)

  return loss


def accuracy(y_top, y, padding_idx):
  _, y_top = y_top.max(-1)
  not_padding = y != padding_idx
  eq = y_top[not_padding] == y[not_padding]

  return eq
