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
import inference
from utils import success, warning, danger


def padded_batch(batch_size, dataset, mode):
  g = dataset.gen(mode)

  while True:
    xs, ys = [], []

    max_x_len = 0
    max_y_len = 0
    total_size = 0

    while len(xs) < batch_size and total_size < 4000:
      x, y = next(g)

      if len(x) > 200 or len(y) > 200:
        continue

      xs.append(x)
      ys.append(y)

      max_x_len = max(max_x_len, len(x))
      max_y_len = max(max_y_len, len(y))
      total_size = max_x_len * len(xs) + max_y_len * len(ys)

    # if total_size > 4000 and len(xs) % 2 == 0:
    #   print('batch truncated: batch_size: {}, max_x_len: {}, max_y_len: {}'.
    #         format(len(xs), max_x_len, max_y_len))
    #   break

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
      "--steps", help="number of steps", type=int, default=1000)
  parser.add_argument(
      "--log-interval", help="log interval", type=int, default=100)
  parser.add_argument(
      "--learning-rate", help="learning rate", type=float, default=0.001)
  parser.add_argument(
      "--dropout", help="dropout probability", type=float, default=0.2)
  parser.add_argument(
      "--pe-type",
      help="positional encoding type",
      type=str,
      choices=['projection', 'addition'],
      default='projection')

  return parser


def main():
  # TODO: try lowercase everything
  # TODO: visualize attention
  # TODO: inference
  # TODO: beam search
  # TODO: add test set
  # TODO: add multi-gpu
  # TODO: try mask attention
  # TODO: split batch on gpus
  # TODO: async
  # TODO: requirements.txt file
  # TODO: attention: in decoder self attention only attend to previous values
  # TODO: try attention padding mask
  # TODO: infer prediction should be the same as eval prediction
  # TODO: train and test: loss, accuracy calculation refactor

  parser = make_parser()
  args = parser.parse_args()
  batch_size = args.batch_size

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

  if args.cuda:
    if torch.cuda.device_count() > 1:
      print(warning('using {} GPUs'.format(torch.cuda.device_count())))
      model = nn.DataParallel(model)
      batch_size = batch_size * torch.cuda.device_count()

    model = model.cuda()

  if os.path.exists(args.weights):
    print(warning('loading state from'), args.weights)
    base_model.load_state_dict(torch.load(args.weights))

  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

  model.train()
  for i, (x, y) in zip(
      range(args.steps),
      padded_batch(batch_size, dataset, 'train'),
  ):
    optimizer.zero_grad()

    x, y = Variable(x), Variable(y)
    if args.cuda:
      x, y = x.cuda(), y.cuda()
    y_bottom, y = y[:, :-1], y[:, 1:]

    y_top = model(x, y_bottom)
    loss = transformer.loss(y_top=y_top, y=y, padding_idx=dataset.pad)
    loss.backward()
    optimizer.step()
    print(danger(i), end='\r')

    if i % args.log_interval == 0:
      model.eval()

      loss = 0
      accuracy = 0
      n_samples = 0
      for j, (x, y) in zip(
          range(batch_size * 10),  # TODO: compute on all test set
          padded_batch(batch_size, dataset, 'tst2012'),
      ):
        x, y = Variable(x, volatile=True), Variable(y, volatile=True)
        if args.cuda:
          x, y = x.cuda(), y.cuda()
        y_bottom, y = y[:, :-1], y[:, 1:]

        y_top = model(x, y_bottom)
        l = transformer.loss(
            y_top=y_top, y=y, padding_idx=dataset.pad, reduce=False)
        a = transformer.accuracy(
            y_top=y_top, y=y, padding_idx=dataset.pad, reduce=False)

        assert l.size() == a.size()
        loss += l.data.sum()
        accuracy += a.data.sum()
        n_samples += l.size(0)

        print(danger('eval batch: {}'.format(j)), end='\r')
      print('\r', end='')

      loss /= n_samples
      accuracy /= n_samples

      print(
          success('step: {}, loss: {:.4f}, accuracy: {:.2f}'.format(
              i, loss, accuracy * 100)))

      for k in range(3):
        print(
            warning('true:'),
            dataset.decode_target(y.data[k]).split('</s>')[0])
        print(
            warning('pred:'),
            dataset.decode_target(torch.max(
                y_top, dim=-1)[1].data[k]).split('</s>')[0])

      print(success('inference:'))
      start = Variable(torch.LongTensor([[1]]) * dataset.sos)
      if args.cuda:
        start = start.cuda()
      inf = inference.Inferer(model)(x[:1], y_bottom=start, max_len=100)
      print(
          warning('true:'),
          dataset.decode_target(y.data[0]).split('</s>')[0])
      print(
          warning('pred:'),
          dataset.decode_target(inf.data[0]).split('</s>')[0])

      torch.save(base_model.state_dict(), args.weights)
      model.train()
      print(warning('model saved to'), args.weights)


if __name__ == '__main__':
  main()
