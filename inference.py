import torch


class Inferer(object):
  def __init__(self, model):
    self.model = model

  def __call__(self, x, y_bottom, max_len):
    return self.infer(x, y_bottom, max_len)

  def infer(self, x, y_bottom, max_len):
    while y_bottom.size(1) < max_len:
      y_top = self.model(x, y_bottom)
      y_top = y_top.max(-1)[1]
      y_bottom = torch.cat([y_bottom, y_top[:, -1:]], -1)

    return y_top

  # def infer(self, x, start, max_len):
  #   y_bottom = start
  #
  #   while y_bottom.size(1) < max_len:
  #     y_top = self.model(x, y_bottom)
  #     y_top = y_top.max(-1)[1]
  #     y_bottom = torch.cat([start, y_top], -1)
  #
  #   return y_top
