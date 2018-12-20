import os
import torch.utils.data


class Vocab(object):
    def __init__(self, path, lang):
        with open(os.path.join(path, 'vocab.{}'.format(lang))) as f:
            vocab = f.read().splitlines()
            vocab = ['<p>'] + vocab

        self.sym2id = {sym: id for id, sym in enumerate(vocab)}
        self.id2sym = {id: sym for id, sym in enumerate(vocab)}

    def __len__(self):
        return len(self.sym2id)

    @property
    def sos_id(self):
        return self.sym2id['<s>']

    @property
    def eos_id(self):
        return self.sym2id['</s>']

    def encode(self, syms):
        return [self.sym2id[sym] if sym in self.sym2id else self.sym2id['<unk>'] for sym in syms]

    def decode(self, ids):
        return [self.id2sym[id] for id in ids]


class TrainEvalDataset(torch.utils.data.Dataset):
    def __init__(self, path, subset, source, target):
        self.data = self.load_data(path, subset, source, target)
        self.source_vocab = Vocab(path, source)
        self.target_vocab = Vocab(path, target)

    def __getitem__(self, item):
        x, y = self.data[item]
        x = [self.source_vocab.sos_id, *self.source_vocab.encode(x), self.source_vocab.eos_id]
        y = [self.target_vocab.sos_id, *self.target_vocab.encode(y), self.target_vocab.eos_id]

        return x, y

    def __len__(self):
        return len(self.data)

    def load_data(self, path, subset, source, target):
        data = []
        source_path = os.path.join(path, '{}.{}'.format(subset, source))
        target_path = os.path.join(path, '{}.{}'.format(subset, target))
        with open(source_path) as f_source, open(target_path) as f_target:
            for x, y in zip(f_source, f_target):
                x, y = x.strip().split(' '), y.strip().split(' ')
                if x[-1] == '.': x = x[:-1]  # TODO: fix this
                if y[-1] == '.': y = y[:-1]  # TODO: fix this

                # TODO:
                if len(x) > 160:
                    continue
                if len(y) > 160:
                    continue

                data.append((x, y))

        return data
