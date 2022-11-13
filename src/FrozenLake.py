import numpy as np
import unittest


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

class Grid:
  def __init__(self,w,h):
    self.w = w
    self.h = h
    self.bad = set()
    
  def is_valid(self, p):
    return 0 <= p[0] < self.h and 0 <= p[1] < self.w and p not in self.bad
  
  def add_bad(self, p):
    self.bad.add(p)

  def to_space_label(self, p):
    return (p[0] * self.w) + p[1]
  
  def to_p(self, s):
    r = s // len(self)
    c = s % len(self)
    return (r,c)
class GridMDP:

  def __init__(self,
               w,
               h,
               motion_forward_prob=1,
               motion_side_prob=0,
               motion_back_prob=0):
    
    self.w = w
    self.h = h
    assert (motion_forward_prob + (2 * motion_side_prob) +
            motion_back_prob) == 1
    self.grid = Grid(w,h)
    self.dirs = Directions()
    self.dirs.set_probabilities(f = motion_forward_prob, s = motion_side_prob, b = motion_back_prob )
    self.T = {}
    # self.compute_state_transition_matrix()

  def empty_states(self, val=None):
    ret = {}
    for r in range(self.h):
      for c in range(self.w):
        if val != None:
          ret[(r, c)] = val
        else:
          ret[(r, c)] = {}
    return ret

  def compute_state_transition_matrix(self, full=True):
    for r in range(self.h):
      for c in range(self.w):
        self.T[(r, c)] = [None for i in range(len(self.dirs))]
        for i in range(len(self.dirs)):
          self.T[(r, c)][i] = self.empty_states(0)
          destinations = self.dirs.get_coordinate_and_prob_from_direction(p=(r,c), direction = i)
          for dest, prob in destinations:
            if self.grid.is_valid(p=dest):
              self.T[(r,c)][i][dest]+=prob
            else:
              #we don't move
              self.T[(r,c)][i][(r,c)]+=prob

  

class TestDirections(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    self.d = Directions()
    self.d.set_probabilities(f=0.8,s=0.1,b=0)
    super(TestDirections, self).__init__(*args, **kwargs)
  def test_up(self):
    self.assertEqual(self.d.get_coordinate_from_direction(p=(10,5), direction=0), (9,5))

  def test_right(self):
    self.assertEqual(self.d.get_coordinate_from_direction(p=(10,5), direction=1), (10,6))

  def test_right_prob(self):
    res = self.d.get_coordinate_and_prob_from_direction(p=(10,5), direction=0)
    for p,prob in res:
      print("p", p, "prob", prob)
if __name__ == '__main__':

  # unittest.main()

  mdp = GridMDP(w=20,
                      h=20,
                      motion_forward_prob=0.8,
                      motion_side_prob=0.1,
                      motion_back_prob=0)
  mdp.grid.add_bad((1,1))
  mdp.compute_state_transition_matrix()
  
  # print(mdp.T[(1,0)][0])
  # print(mdp.T[(0,0)][0])