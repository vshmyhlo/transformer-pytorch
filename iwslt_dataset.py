import os

# def gen(dir, dataset):
#   with open(os.path.join(dir, dataset + '.en')) as f_en, open(
#       os.path.join(dir, dataset + '.vi')) as f_vi:
#     for row_en, row_vi in zip(f_en, f_vi):
#       row_en, row_vi = row_en.split(' '), row_vi.split(' ')
#       print(row_en)
#       print(row_vi)
#       fail

# yield 1, 2

# def build_sym2id(id2sym):
#   return {sym: id for id, sym in enumerate(id2sym)}


def encode(syms, sym2id):
  return [sym2id[sym] if sym in sym2id else sym2id['<unk>'] for sym in syms]


def decode(ids, id2sym):
  return ' '.join([id2sym[id] for id in ids])


def make_vocab(f):
  vocab = f.read().splitlines()
  vocab = ['<p>'] + vocab

  sym2id = {sym: id for id, sym in enumerate(vocab)}
  id2sym = {id: sym for id, sym in enumerate(vocab)}

  return sym2id, id2sym


class Dataset(object):
  def __init__(self, dir, source, target):
    self.dir = dir
    self.source = source
    self.target = target
    self.init_vocabs()

  def init_vocabs(self):
    with open(os.path.join(self.dir, 'vocab.' + self.source)) as f:
      self.source_sym2id, self.source_id2sym = make_vocab(f)
    with open(os.path.join(self.dir, 'vocab.' + self.target)) as f:
      self.target_sym2id, self.target_id2sym = make_vocab(f)

    for name, sym in [('pad', '<p>'), ('sos', '<s>'), ('eos', '</s>')]:
      assert self.source_sym2id[sym] == self.target_sym2id[sym]
      setattr(self, name, self.source_sym2id[sym])

    self.source_vocab_size = len(self.source_id2sym)
    self.target_vocab_size = len(self.target_id2sym)

  def gen(self, mode):
    source_path = os.path.join(self.dir, mode + '.' + self.source)
    target_path = os.path.join(self.dir, mode + '.' + self.target)

    with open(source_path) as f_source, open(target_path) as f_target:
      for x, y in zip(f_source, f_target):
        x, y = x.strip().split(' '), y.strip().split(' ')
        if x[-1] == '.': x = x[:-1]
        if y[-1] == '.': y = y[:-1]
        x, y = self.encode_source(x), self.encode_target(y)
        yield x, y

  def encode_source(self, syms):
    return encode(syms, self.source_sym2id)

  def encode_target(self, syms):
    return encode(syms, self.target_sym2id)

  def decode_source(self, ids):
    return decode(ids, self.source_id2sym)

  def decode_target(self, ids):
    return decode(ids, self.target_id2sym)


def main():
  ds = Dataset('./iwslt15', 'train', source='vi', target='en')
  print('vi: {}'.format(ds.source_vocab_size))
  print('en: {}'.format(ds.target_vocab_size))
  g = ds.gen()

  for i in range(3):
    x, y = next(g)
    print('x: {}\ny: {}'.format(ds.decode_source(x), ds.decode_target(y)))


if __name__ == '__main__':
  main()
