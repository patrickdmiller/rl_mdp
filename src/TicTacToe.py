import numpy as np

class TicTacToeGrid:
  def __init__(self, n):
    self.n = n
    self.m = np.zeros([n,n])

    print(self.m)
    print(np.rot90(self.m))
    print(bytes(self.m))

  def state_key(self):
    hashes = [hash(bytes(self.m))]
    _m = self.m
    for i in range(3):
      _m = np.rot90(_m)
      hashes.append(hash(bytes(_m)))
    print(hashes)

class TicTacToe:
  
  def __init__(self, n):
    self.n = n

  # def build_states(self):

if __name__ == '__main__':
  g = TicTacToeGrid(5)
  g.state_key()