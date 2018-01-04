import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import python_format as dataset
from transformer import Tranformer


def gen(batch_size):
  g = dataset.gen(min_len=3, max_len=4)

  while True:
    xs, ys = [], []

    for i in range(batch_size):
      x, y = next(g)
      xs.append(x)
      ys.append(y)

    x_len = [len(x) for x in xs]
    y_len = [len(y) for y in ys]

    x = [x + [dataset.eos] + [dataset.pad] * (max(x_len) - len(x)) for x in xs]
    y = [y + [dataset.eos] + [dataset.pad] * (max(y_len) - len(y)) for y in ys]

    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    yield (x, y)


def main():
  steps = 1000
  log_interval = 10

  model = Tranformer(
      source_vocab_size=dataset.vocab_size,
      target_vocab_size=dataset.vocab_size,
      size=128,
      num_layers=2)

  optimizer = optim.Adam(model.parameters(), lr=0.001)

  model.train()
  for i, (x, y) in zip(range(steps), gen(32)):
    optimizer.zero_grad()

    x, y = Variable(x), Variable(y)
    y_bottom, y = y[:, :-1], y[:, 1:]

    y_top = model(x, y_bottom)

    loss = F.cross_entropy(
        y_top.view(-1, dataset.vocab_size),
        y.contiguous().view(-1))

    loss.backward()
    optimizer.step()

    if i % log_interval == 0:
      print('step: {}, loss: {:4f}\n\ttrue: {}\n\tpred: {}\n'.format(
          i,
          loss.data[0],
          dataset.decode(y.data[0]),
          dataset.decode(torch.max(y_top, dim=-1)[1].data[0]),
      ))


if __name__ == '__main__':
  main()
