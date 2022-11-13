'''
author: patrick miller
gihub: github.com/patrickdmiller
gatech: pmiller75
'''

import unittest
import numpy as np

#0 is up, 1 is right
class Directions:
  def __init__(self):
    self.dir_transitions = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    self.probs = {}
    self.length = 4
  def get_coordinate_from_direction(self, p, direction):
    assert direction < len(self.dir_transitions) and direction >= 0
    return (p[0] + self.dir_transitions[direction][0],
                 p[1] + self.dir_transitions[direction][1])

  def set_probabilities(self, f=1, s=0, b=0):
    self.probs['f'] = f
    self.probs['s'] = s
    self.probs['b'] = b
  
  #return coordinates and probabilities
  def get_coordinate_and_prob_from_direction(self, p, direction):
    ret = []
    if self.probs['f'] > 0:
      ret.append((self.get_coordinate_from_direction(p, direction), self.probs['f']))
    if self.probs['s'] > 0:
      #ccw
      ccw_direction = (direction - 1) % len(self)
      ret.append((self.get_coordinate_from_direction(p, ccw_direction), self.probs['s']))
      #cw
      cw_direction = (direction + 1) % len(self)
      ret.append((self.get_coordinate_from_direction(p, cw_direction), self.probs['s']))
    if self.probs['b'] > 0:
      ret.append((self.get_coordinate_from_direction(p, (direction + 2) % len(self)), self.probs['b']))
    return ret
  
  def __len__(self):
    return self.length
class Strategy:
  def __init__(self):
    pass

  
class Grid:
  def __init__(self,w,h):
    self.w = w
    self.h = h
    self.invalid = set()
    self.rewards = {} #(r,c):amount
    
  def is_valid(self, p):
    return 0 <= p[0] < self.h and 0 <= p[1] < self.w and p not in self.invalid
  
  def add_invalid(self, p):
    self.invalid.add(p)
  def add_reward(self, p, amt):
    self.rewards[p] = amt
  def to_s(self, p):
    return (p[0] * self.w) + p[1]
  
  def to_p(self, s):
    r = s // self.w
    c = s % self.w
    return (r,c)
class GridMDP:

  def __init__(self,
               w,
               h,
               motion_forward_prob=1,
               motion_side_prob=0,
               motion_back_prob=0):
    
    # self.w = w
    # self.h = h
    assert (motion_forward_prob + (2 * motion_side_prob) +
            motion_back_prob) == 1
    self.grid = Grid(w,h)
    self.dirs = Directions()
    self.dirs.set_probabilities(f = motion_forward_prob, s = motion_side_prob, b = motion_back_prob )
    self.P_raw = {}
    self.P = self.P_raw
    self.Policy = self.empty_states(-1)
    self.S = self.build_statemap()
    # self.compute_state_transition_matrix()
  def build_statemap(self):
    pass
  
  def build_policy(self, strategy=None):
    if not strategy:
      #random
      for r in range(self.grid.h):
        for c in range(self.grid.w):
          self.Policy[(r,c)] = np.random.randint(0, len(self.dirs))
          

  def get_action(self, p = None, s = None):
    as_p = True
    if p is None:
      as_p = False
      p = self.grid.to_p(s)
    
    action = self.Policy[p]
    return action
    if as_p:
      return action
    else:
      return self.grid.to_s(action)
  
  def empty_states(self, val=None):
    ret = {}
    for r in range(self.grid.h):
      for c in range(self.grid.w):
        if val != None:
          ret[(r, c)] = val
        else:
          ret[(r, c)] = {}
    return ret

  def compute_state_transition_matrix(self, full=True):
    for r in range(self.grid.h):
      for c in range(self.grid.w):
        self.P_raw[(r, c)] = [None for i in range(len(self.dirs))]
        for i in range(len(self.dirs)):
          self.P_raw[(r, c)][i] = self.empty_states(0)
          destinations = self.dirs.get_coordinate_and_prob_from_direction(p=(r,c), direction = i)
          for dest, prob in destinations:
            if self.grid.is_valid(p=dest):
              self.P_raw[(r,c)][i][dest]+=prob
            else:
              self.P_raw[(r,c)][i][(r,c)]+=prob

      # self.T.append(self.T_raw[(r,c)])

class TestDirections(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    self.d = Directions()
    self.d.set_probabilities(f=0.8,s=0.1,b=0)
    self.grid = Grid(w=4,h=3)
    
    self.mdp = GridMDP(w=4,
                      h=3,
                      motion_forward_prob=0.8,
                      motion_side_prob=0.1,
                      motion_back_prob=0)
    self.mdp.grid.add_invalid((1,1))
    self.mdp.compute_state_transition_matrix()
    
    super(TestDirections, self).__init__(*args, **kwargs)
  def test_up(self):
    self.assertEqual(self.d.get_coordinate_from_direction(p=(10,5), direction=0), (9,5))

  def test_right(self):
    self.assertEqual(self.d.get_coordinate_from_direction(p=(10,5), direction=1), (10,6))

  def test_right_prob(self):
    res = self.d.get_coordinate_and_prob_from_direction(p=(10,5), direction=0)
    self.assertEqual(res, [((9,5), 0.8), ((10,4), 0.1), ((10,6), 0.1)])

  def test_grid_label(self):
    self.assertEqual(self.grid.to_p(10), (2,2))
    self.assertEqual(self.grid.to_s((2,2)), 10)
  
  def test_t(self):
    p = self.mdp.grid.to_p(8)
    self.assertEqual(self.mdp.P[p][2][self.mdp.grid.to_p(8)], 0.9)
    self.assertEqual(self.mdp.P[p][2][self.mdp.grid.to_p(9)], 0.1)
    #test bad boundary, should hit and go back so P(1') == .8
    self.assertEqual(self.mdp.P[self.mdp.grid.to_p(1)][2][self.mdp.grid.to_p(1)], 0.8)
if __name__ == '__main__':

  unittest.main()

  mdp = GridMDP(w=4,
                      h=3,
                      motion_forward_prob=0.8,
                      motion_side_prob=0.1,
                      motion_back_prob=0)
  grid = mdp.grid
  grid.add_invalid((1,1))
  grid.add_reward(grid.to_p(3), 1)
  grid.add_reward(grid.to_p(7), -1)
  mdp.compute_state_transition_matrix()
  mdp.build_policy()
  print(mdp.Policy)
  print(mdp.get_action(s=0))


