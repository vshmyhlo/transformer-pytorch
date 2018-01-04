import string
import numpy as np
from faker import Faker

faker = Faker('en_US')

special_symbols = {
    '<p>': 0,
    '<s>': 1,
    '</s>': 2,
    '<f>': 3,
    '<n>': 4,
}

vocab = string.ascii_letters + '0123456789 %-:{}'

sym2id = {sym: i + len(special_symbols) for i, sym in enumerate(vocab)}
sym2id = {**special_symbols, **sym2id}
id2sym = {sym2id[sym]: sym for sym in sym2id}

pad = sym2id['<p>']
sos = sym2id['<s>']
eos = sym2id['</s>']

vocab_size = len(vocab) + len(special_symbols)
assert vocab_size == len(sym2id) and vocab_size == len(id2sym)

word_styles = [
    str.lower,
    str.lower,
    str.lower,
    str.title,
    str.title,
    str.upper,
]


def encode(template, subs):
  input = list(template) + ['<f>']
  for sub in subs:
    input += list(sub) + ['<n>']
  input = input[:-1]
  output = list(template.format(*subs))
  return [sym2id[x] for x in input], [sym2id[x] for x in output]


def decode(ids):
  return ''.join([id2sym[id] for id in ids])


def sample_num():
  num = str(np.random.randint(1000))
  return '{}', [num]


def sample_word():
  word = faker.word()
  style = np.random.choice(word_styles)
  return '{}', [style(word)]


def sample_percent():
  text, subs = sample_num()
  return text + '%', subs


def sample_nth():
  text, subs = sample_num()
  return text + '-th', subs


def sample_word_colon_num():
  word, word_subs = sample_word()
  num, num_subs = sample_num()
  return word + ': ' + num, word_subs + num_subs


def sample_part():
  sample = np.random.choice([
      sample_word,
      sample_word,
      sample_word,
      sample_word,
      sample_num,
      sample_percent,
      sample_nth,
      sample_word_colon_num,
  ])

  text, subs = sample()
  replace = np.random.choice([False, False, True])

  if replace:
    return text.format(*subs), []
  else:
    return text, subs


def sample(min_len, max_len):
  template = []
  subs = []

  for i in range(np.random.randint(min_len, max_len)):
    t, s = sample_part()
    template.append(t)
    subs += s

  return ' '.join(template), subs


def gen(min_len, max_len):
  while True:
    x, y = encode(*sample(min_len, max_len))
    yield x, y


def main():
  g = gen(1, 7)

  for i in range(3):
    x, y = next(g)
    print('x: {}\ny: {}'.format(decode(x), decode(y)))


if __name__ == '__main__':
  main()
