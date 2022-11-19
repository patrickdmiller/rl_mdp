import numpy as np
from hashlib import sha1
import unittest


class TicTacToeState:
  def __init__(self, n):
    self.n = n
    self.m = np.zeros([n,n], dtype=np.int8)
    self.rows = [0 for i in range(n)]
    self.cols = [0 for i in range(n)]
    self.diag_d = 0
    self.diag_u = 0
    
  def hash(self, a):
    return ''.join(list(map(lambda c:''.join(list(map(lambda x: str(int(x)), c))), a)))
  
  def do_move(self, r,c, piece):
    self.m[r,c] = piece
    self.rows[r]+=piece
    self.cols[c]+=piece
    if r == c: 
      self.diag_d+=piece
    if r + c == self.n-1:
      self.diag_u+=piece
    nn = self.n
    if self.rows[r] == nn or self.cols[c]==nn or self.diag_u == nn or self.diag_d == nn:
      return True
    nn = -nn
    if self.rows[r] == nn or self.cols[c]==nn or self.diag_u == nn or self.diag_d == nn:
      return True
    return False
      
  def clear_move(self, r, c):
    piece = self.m[r,c] * -1
    self.m[r,c] = 0
    self.rows[r]+=piece
    self.cols[c]+=piece
    if r == c: 
      self.diag_d+=piece
    if r + c == self.n-1:
      self.diag_u+=piece
    
    
  def state_key(self):
    _m = self.m
    hashes = [self.hash(_m)]
    for i in range(3):
      _m = np.rot90(_m)
      hashes.append(self.hash(_m))
    hashes.sort()
    return hashes[0]
  def __repr__(self):
    return str(self.m)
  # def copy
class TicTacToe:
  
  def __init__(self, n):
    self.n = n

  def build_state_space(self):
    
    states = set()
    state = TicTacToeState(n = self.n)
    def build(state, piece):
      print(state)
      for r in range(self.n):
        for c in range(self.n):
          if state.m[r,c] == 0:
            #add it
            state.m[r,c] = piece
            is_win = state.do_move(r,c,piece)
            if is_win:
              print("winner: ", state.state_key(), state)
            else:
              if state.state_key() not in states:
                states.add(state.state_key())
                build(state=state, piece=piece*-1)
                #remove it

            state.clear_move(r,c)
                # break
    build(state,1)
    print(states)
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
  # unittest.main()
  t = TicTacToe(3)
  t.build_state_space()