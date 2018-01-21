import os
import pickle
from termcolor import colored


class PersistentDict(object):
  def __init__(self, path):
    self.path = path

    if os.path.exists(path):
      with open(path, 'rb') as f:
        self.data = pickle.load(f)
    else:
      self.data = {}

    print('len(self.data)', len(self.data))

  def __contains__(self, key):
    return self.data.__contains__(key)

  def __getitem__(self, key):
    return self.data.__getitem__(key)

  def __setitem__(self, key, value):
    res = self.data.__setitem__(key, value)

    with open(self.path, 'wb') as f:
      pickle.dump(self.data, f)

    return res


def success(str):
  return colored(str, 'green')


def warning(str):
  return colored(str, 'yellow')


def danger(str):
  return colored(str, 'red')
