from collections import defaultdict

# https://stackoverflow.com/a/54790316
def defaultdict_gen(n):
  if n < 1:
    raise ValueError()
  if n == 1:
    return defaultdict(dict)
  return defaultdict(lambda: defaultdict_gen(n - 1))