import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import python_format_dataset as dataset
import iwslt_dataset
import transformer


def padded_batch(batch_size, dataset, mode):
  g = dataset.gen(mode)

  while True:
    xs, ys = [], []

    max_x_len = 0
    max_y_len = 0

    for i in range(batch_size):
      x, y = next(g)
      xs.append(x)
      ys.append(y)

      max_x_len = max(max_x_len, len(x))
      max_y_len = max(max_y_len, len(y))
      # TODO: async

      total_size = max_x_len * len(xs) + max_y_len * len(ys)
      if total_size > 4000 and len(xs) % 2 == 0:  # TODO: size per gpu
        print('batch truncated: batch_size: {}, max_x_len: {}, max_y_len: {}'.
              format(len(xs), max_x_len, max_y_len))
        break

    x = [[dataset.sos] + x + [dataset.eos] + [dataset.pad] *
         (max_x_len - len(x)) for x in xs]
    y = [[dataset.sos] + y + [dataset.eos] + [dataset.pad] *
         (max_y_len - len(y)) for y in ys]

    x = torch.LongTensor(x)
    y = torch.LongTensor(y)

    yield (x, y)


def make_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--weights", help="weight file", type=str, required=True)
  parser.add_argument("--batch-size", help="batch size", type=int, default=32)
  parser.add_argument("--size", help="transformer size", type=int, default=256)
  parser.add_argument("--cuda", help="use cuda", action='store_true')
  parser.add_argument(
      "--n-layers", help="number of transformer layers", type=int, default=4)
  parser.add_argument(
      "--n-heads", help="number of transformer heads", type=int, default=4)
  parser.add_argument(
      "--steps", help="number of steps", type=int, default=1000)
  parser.add_argument(
      "--log-interval", help="log interval", type=int, default=100)
  parser.add_argument(
      "--learning-rate", help="learning rate", type=float, default=0.001)
  parser.add_argument(
      "--dropout", help="dropout probability", type=float, default=0.2)

  return parser


def main():
  # TODO: visualize attention
  # TODO: inference
  # TODO: beam search
  # TODO: add test set
  # TODO: add multi-gpu
  # TODO: try mask attention
  # TODO: split batch on gpus

  parser = make_parser()
  args = parser.parse_args()

  dataset = iwslt_dataset.Dataset('./iwslt15', source='vi', target='en')
  model = transformer.Tranformer(
      source_vocab_size=dataset.source_vocab_size,
      target_vocab_size=dataset.target_vocab_size,
      size=args.size,
      n_layers=args.n_layers,
      n_heads=args.n_heads,
      dropout=args.dropout,
      padding_idx=dataset.pad)

  if args.cuda:
    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs")
      model = nn.DataParallel(model)

    model = model.cuda()

  if os.path.exists(args.weights):
    model.load_state_dict(torch.load(args.weights))

  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

  model.train()
  for i, (x, y) in zip(
      range(args.steps),
      padded_batch(args.batch_size, dataset, 'train'),
  ):
    optimizer.zero_grad()

    x, y = Variable(x), Variable(y)
    if args.cuda:
      if torch.cuda.device_count() > 1:
        x, y = nn.DataParallel(x), nn.DataParallel(y)

      x, y = x.cuda(), y.cuda()

    y_bottom, y = y[:, :-1], y[:, 1:]

    y_top = model(x, y_bottom)
    loss = transformer.loss(y_top=y_top, y=y)
    accuracy = transformer.accuracy(y_top=y_top, y=y)

    if i % args.log_interval == 0:
      print(
          'step: {}, loss: {:.4f}, accuracy: {:.2f}\n\ttrue: {}\n\tpred: {}\n'.
          format(
              i,
              loss.data[0],
              accuracy.data[0] * 100,
              dataset.decode_target(y.data[0]),
              dataset.decode_target(torch.max(y_top, dim=-1)[1].data[0]),
          ))

      torch.save(model.state_dict(), args.weights)

    loss.backward()
    optimizer.step()


if __name__ == '__main__':
  main()
