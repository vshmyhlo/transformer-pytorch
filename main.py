import random
import os
import itertools
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import iwslt_dataset
import transformer
import inference
import metrics
from utils import success, warning, danger, log_args, PersistentDict


def sorted_gen(dataset, mode):
  for x, y in sorted(
      dataset.gen(mode),
      key=lambda xy: max(len(xy[0]), len(xy[1])),
      reverse=True,
  ):
    yield x, y


def shuffle(gen):
  seq = list(gen)
  random.shuffle(seq)
  for x in seq:
    yield x


def padded_batch(batch_size, dataset, mode, n_devices, batch2batch_size):
  g = sorted_gen(dataset, mode)

  for batch_i in itertools.count():
    x, y = next(g)
    max_x_len = len(x)
    max_y_len = len(y)
    xs, ys = [x], [y]

    if batch_i in batch2batch_size:
      real_batch_size = batch2batch_size[batch_i]
    else:
      real_batch_size = batch_size

    while len(xs) < (real_batch_size * n_devices):
      x, y = next(g)

      max_x_len = max(max_x_len, len(x))
      max_y_len = max(max_y_len, len(y))

      xs.append(x)
      ys.append(y)

    x = [[dataset.sos] + x + [dataset.eos] + [dataset.pad] *
         (max_x_len - len(x)) for x in xs]
    y = [[dataset.sos] + y + [dataset.eos] + [dataset.pad] *
         (max_y_len - len(y)) for y in ys]

    x = torch.LongTensor(x)
    y = torch.LongTensor(y)

    batch2batch_size[batch_i] = real_batch_size

    yield batch_i, (x, y)


def make_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--weights", help="weight file", type=str, required=True)
  parser.add_argument("--batch-size", help="batch size", type=int, default=32)
  parser.add_argument("--size", help="transformer size", type=int, default=256)
  parser.add_argument("--cuda", help="use cuda", action='store_true')
  parser.add_argument(
      "--epochs", help="number of epochs", type=int, default=10)
  parser.add_argument(
      "--dataset-path", help="dataset folder", type=str, default='./iwslt15')
  parser.add_argument(
      "--source-lng", help="source language", type=str, default='en')
  parser.add_argument(
      "--target-lng", help="target language", type=str, default='vi')
  parser.add_argument(
      "--n-layers", help="number of transformer layers", type=int, default=4)
  parser.add_argument(
      "--n-heads", help="number of transformer heads", type=int, default=4)
  parser.add_argument(
      "--learning-rate", help="learning rate", type=float, default=0.001)
  parser.add_argument(
      "--dropout", help="dropout probability", type=float, default=0.2)
  parser.add_argument(
      "--pe-type",
      help="positional encoding type",
      type=str,
      choices=['projection', 'addition'],
      default='addition')

  return parser


def main():
  # TODO: try lowercase everything
  # TODO: visualize attention
  # TODO: beam search
  # TODO: try mask attention
  # TODO: try attention padding mask
  # TODO: async
  # TODO: requirements.txt file
  # TODO: attention: in decoder self attention only attend to previous values
  # TODO: byte pair encoding
  # TODO: compute bleu (https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)

  parser = make_parser()
  args = parser.parse_args()
  log_args(args)
  batch2batch_size = PersistentDict('./batch2batch_size')
  print(
      warning('PersistentDict: len(data) == {}'.format(
          len(batch2batch_size.data))))
  print(sorted(batch2batch_size.data.values()))

  dataset = iwslt_dataset.Dataset(
      args.dataset_path, source=args.source_lng, target=args.target_lng)
  base_model = transformer.Tranformer(
      source_vocab_size=dataset.source_vocab_size,
      target_vocab_size=dataset.target_vocab_size,
      size=args.size,
      n_layers=args.n_layers,
      n_heads=args.n_heads,
      pe_type=args.pe_type,
      dropout=args.dropout,
      padding_idx=dataset.pad)
  model = base_model

  n_devices = 1
  if args.cuda:
    n_devices = torch.cuda.device_count()
    if n_devices > 1:
      print(warning('using {} GPUs'.format(n_devices)))
      model = nn.DataParallel(model)
    model = model.cuda()

  if os.path.exists(args.weights):
    print(warning('state loaded from'), args.weights)
    base_model.load_state_dict(torch.load(args.weights))

  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

  for epoch in range(args.epochs):
    print(success('epoch: {}'.format(epoch)))

    # Train ####################################################################
    summary = metrics.Summary((0, 0))
    model.train()

    for i, (batch_i, (x, y)) in zip(
        itertools.count(),
        padded_batch(
            args.batch_size,
            dataset,
            mode='train',
            n_devices=n_devices,
            batch2batch_size=batch2batch_size),
    ):
      try:
        optimizer.zero_grad()
      except RuntimeError:
        print(danger('optimizer.zero_grad() failed'))
        optimizer.zero_grad()

      try:
        x, y = Variable(x), Variable(y)
        print(
            danger('train batch {}: x {}, y {}'.format(i, tuple(
                x.size()), tuple(y.size())) + ' ' * 10),
            end='\r')
        if args.cuda:
          x, y = x.cuda(), y.cuda()
        y_bottom, y = y[:, :-1], y[:, 1:]

        y_top = model(x, y_bottom)
        loss = metrics.loss(y_top=y_top, y=y, padding_idx=dataset.pad)
        accuracy = metrics.accuracy(y_top=y_top, y=y, padding_idx=dataset.pad)
        loss.mean().backward()
        optimizer.step()

        summary.add((loss.data, accuracy.data))

      except RuntimeError as e:
        if e.args[0].startswith('cuda runtime error (2) : out of memory'):
          batch2batch_size[batch_i] //= 2
        else:
          raise e

    loss, accuracy = summary.calculate()
    print(
        success('(train) loss: {:.4f}, accuracy: {:.2f}'.format(
            loss, accuracy * 100)))

    # Eval #####################################################################
    summary = metrics.Summary((0, 0))
    model.eval()

    for j, (_, (x, y)) in zip(
        itertools.count(),
        padded_batch(
            32,
            dataset,
            mode='tst2012',
            n_devices=n_devices,
            batch2batch_size={}),
    ):
      x, y = Variable(x, volatile=True), Variable(y, volatile=True)
      print(
          danger('eval batch {}: x {}, y {}'.format(j, tuple(
              x.size()), tuple(y.size())) + ' ' * 10),
          end='\r')
      if args.cuda:
        x, y = x.cuda(), y.cuda()
      y_bottom, y = y[:, :-1], y[:, 1:]

      y_top = model(x, y_bottom)
      loss = metrics.loss(y_top=y_top, y=y, padding_idx=dataset.pad)
      accuracy = metrics.accuracy(y_top=y_top, y=y, padding_idx=dataset.pad)

      summary.add((loss.data, accuracy.data))

    loss, accuracy = summary.calculate()
    print(
        success('(eval) loss: {:.4f}, accuracy: {:.2f}'.format(
            loss, accuracy * 100)))

    for true, pred in zip(y.data[:3], torch.max(y_top, dim=-1)[1].data[:3]):
      print(warning('true:'), dataset.decode_target(true).split('</s>')[0])
      print(warning('pred:'), dataset.decode_target(pred).split('</s>')[0])

    # Infer ####################################################################
    print(success('inference:'))
    inferer = inference.Inferer(model)
    start = Variable(torch.LongTensor(1, 1).fill_(dataset.sos))
    if args.cuda:
      start = start.cuda()
    for true, pred in zip(y.data,
                          inferer(x[:1], y_bottom=start, max_len=100).data):
      print(warning('true:'), dataset.decode_target(true).split('</s>')[0])
      print(warning('pred:'), dataset.decode_target(pred).split('</s>')[0])

    # Saving ###################################################################
    torch.save(base_model.state_dict(), args.weights)
    print(warning('state saved to'), args.weights)


if __name__ == '__main__':
  main()
