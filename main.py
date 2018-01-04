import torch
from torch.autograd import Variable
import python_format as dataset
from transformer import Tranformer


def gen(batch_size):
  g = dataset.gen(min_len=3, max_len=7)

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
  steps = 1
  model = Tranformer(
      num_source_embeddings=dataset.vocab_size,
      num_target_embeddings=dataset.vocab_size,
      size=128)
  train_data = gen(32)

  model.train()
  for i, (x, y) in zip(range(steps), train_data):
    x, y = Variable(x), Variable(y)
    y_bottom, y = y[:, :-1], y[:, 1:]

    y_top = model(x, y_bottom)

    print('out')
    print(y_top.size(), y.size())


if __name__ == '__main__':
  main()
