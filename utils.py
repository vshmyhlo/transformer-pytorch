from termcolor import colored


def success(str):
  return colored(str, 'green')


def warning(str):
  return colored(str, 'yellow')


def danger(str):
  return colored(str, 'red')
