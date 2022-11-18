import numpy as np
from hashlib import sha1
import unittest


class TicTacToeState:
  def __init__(self, n):
    self.n = n
    self.m = np.zeros([n,n], dtype=np.int8)
    
  def hash(self, a):
    return ''.join(list(map(lambda c:''.join(list(map(lambda x: str(int(x)), c))), a)))
    
  def state_key(self):
    _m = self.m
    hashes = [self.hash(_m)]
    for i in range(3):
      _m = np.rot90(_m)
      hashes.append(self.hash(_m))
    hashes.sort()
    return hashes[0]

class TicTacToe:
  
  def __init__(self, n):
    self.n = n

  def build_state_space(self):
    states = {}
    def build():
      for r in range(self.n):
        for c in range(self.n):
          pass

if __name__ == '__main__':
  class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
      super(Test, self).__init__(*args, **kwargs)
  
    def test_grid(self):
      g = TicTacToeState(3)
      g.m[0,2] = 1
      g2 = TicTacToeState(3)
      g2.m[2,2] =1
      self.assertEqual(g.state_key(), g2.state_key())
      g2.m[0,2]=1
      self.assertNotEqual(g.state_key(), g2.state_key())
  unittest.main()