'''
author: patrick miller
gihub: github.com/patrickdmiller
gatech: pmiller75
'''

import unittest
import numpy as np

#0 is up, 1 is right
class Directions:
  def __init__(self, grid):
    self.dir_transitions = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    self.probs = {}
    self.length = 4
    self.grid = grid

  def get_coordinate_from_direction(self, p, direction):
    assert direction < len(self.dir_transitions) and direction >= 0
    return (p[0] + self.dir_transitions[direction][0],
                 p[1] + self.dir_transitions[direction][1])

  def set_probabilities(self, f=1, s=0, b=0):
    self.probs['f'] = f
    self.probs['s'] = s
    self.probs['b'] = b
  
  #return coordinates and probabilities that we would try and go to
  def get_target_coordinate_and_prob_from_direction(self, p, direction):
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
  
  #return actual coordinates and probabilities that we would go to
  def get_resulting_coordinate_and_prob_from_direction(self, p, direction):
    results = []
    targets = self.get_target_coordinate_and_prob_from_direction( p, direction)
    
    # [((2, 4), 0.8), ((1, 3), 0.1), ((3, 3), 0.1)]
    for coord, prob in targets:
      if not self.grid.is_valid(coord):
        results.append((p, prob))
      else:
        results.append((coord, prob))
    return results
  def __len__(self):
    return self.length

class PolicyStrategy:
  def __init__(self):
    pass

class PolicyIterationStrategy(PolicyStrategy):
  def __init__(self, grid, policy, dirs, gamma=0.5): #inject grid and policy
    self.grid = grid
    self.policy = policy
    self.dirs = dirs
    self.gamma = gamma
  def evaluate(self):
    #iterate over each 
    utilities = self.grid.empty_states(0)
    for r in range(self.grid.h):
      for c in range(self.grid.w):
        u = self.grid.get_reward((r,c))
        # print(u)
        if not self.grid.is_terminal((r,c)) and self.grid.is_valid((r,c)):
          print("not terminal: ", (r,c))
          results = self.dirs.get_resulting_coordinate_and_prob_from_direction(p=(r,c), direction=self.policy[(r,c)])
          for p, prob in results:
            u += (self.gamma * prob * utilities[p])
        utilities[(r,c)] = u
    print(utilities)
    
  def generate(self):
    for r in range(self.grid.h):
      for c in range(self.grid.w):
        if self.grid.is_valid((r,c)) and not self.grid.is_bad((r,c)):
          self.policy[(r,c)] = np.random.randint(0, len(self.dirs))


  
class Grid:
  def __init__(self,w,h, default_reward=0, default_bad_reward=-99):
    self.w = w
    self.h = h
    self.invalid = set()
    self.bad = set() #bad is like a hole in frozen lake. invalid is like a wall. 
    self.terminal = set()
    
    self.rewards = {} #(r,c):amount
    
    self.default_reward = default_reward
    self.default_bad_reward = default_bad_reward
    
  def is_valid(self, p):
    return 0 <= p[0] < self.h and 0 <= p[1] < self.w and p not in self.invalid
  
  def is_bad(self, p):
    return p in self.bad
  
  def is_terminal(self, p):
    return self.is_bad(p) or p in self.terminal
  
  def add_terminal(self, p):
    self.terminal.add(p)
    
  def add_invalid(self, p):
    self.invalid.add(p)
  
  def add_bad(self, p):
    self.bad.add(p) 
    self.add_reward(p, self.default_bad_reward)
    
  def get_reward(self, p):
    if p in self.rewards:
      return self.rewards[p]
    if self.is_valid(p):
      return self.default_reward  
    return 0
  
  def add_reward(self, p, amt):
    self.rewards[p] = amt
  
  def to_s(self, p):
    return (p[0] * self.w) + p[1]
  
  def to_p(self, s):
    r = s // self.w
    c = s % self.w
    return (r,c)
  
  def empty_states(self, val=None):
    ret = {}
    for r in range(self.h):
      for c in range(self.w):
        if val != None:
          ret[(r, c)] = val
        else:
          ret[(r, c)] = {}
    return ret
  
class GridMDP:

  def __init__(self,
               w,
               h,
               motion_forward_prob=1,
               motion_side_prob=0,
               motion_back_prob=0,
               default_reward = 0,
               grid = None):
    
    # self.w = w
    # self.h = h
    assert (motion_forward_prob + (2 * motion_side_prob) +
            motion_back_prob) == 1
    self.grid = grid
    if grid is None:
      self.grid = Grid(w=w, h=h, default_reward=default_reward )
    self.dirs = Directions(grid=self.grid)
    self.dirs.set_probabilities(f = motion_forward_prob, s = motion_side_prob, b = motion_back_prob )
    self.P = {}
    # self.P = self.P
    self.policy = self.grid.empty_states(-1)
    self.S = self.build_statemap()
    # self.compute_state_transition_matrix()
  def build_statemap(self):
    pass
  
  def build_policy(self, strategy=None, starting_policy=None):
    if not strategy:
      #random
      for r in range(self.grid.h):
        for c in range(self.grid.w):
          self.policy[(r,c)] = np.random.randint(0, len(self.dirs))
    else:
      self.policy = starting_policy
      self.strategy = strategy(grid = self.grid, policy = self.policy, dirs = self.dirs)
      if starting_policy is None:
        self.strategy.generate()
    
      self.strategy.evaluate()

  def get_action(self, p = None, s = None):
    as_p = True
    if p is None:
      as_p = False
      p = self.grid.to_p(s)
    
    action = self.policy[p]
    return action


  def compute_state_transition_matrix(self, full=False):
    if full:
      for r in range(self.grid.h):
        for c in range(self.grid.w):
          self.P[(r, c)] = [None for i in range(len(self.dirs))]
          for i in range(len(self.dirs)):
            self.P[(r, c)][i] = self.grid.empty_states(0)
            destinations = self.dirs.get_target_coordinate_and_prob_from_direction(p=(r,c), direction = i)
            for dest, prob in destinations:
              if self.grid.is_valid(p=dest):
                self.P[(r,c)][i][dest]+=prob
              else:
                self.P[(r,c)][i][(r,c)]+=prob
    if not full:
      for r in range(self.grid.h):
        for c in range(self.grid.w):
          self.P[(r, c)] = [None for i in range(len(self.dirs))]
          for i in range(len(self.dirs)):
            self.P[(r, c)][i] = {}
            destinations = self.dirs.get_target_coordinate_and_prob_from_direction(p=(r,c), direction = i)
            for dest, prob in destinations:
              if self.grid.is_valid(p=dest):
                if dest not in self.P[(r,c)][i]:
                  self.P[(r,c)][i][dest] = 0
                self.P[(r,c)][i][dest]+=prob
              else:
                if (r,c) not in self.P[(r,c)][i]:
                  self.P[(r,c)][i][(r,c)] = 0
                self.P[(r,c)][i][(r,c)]+=prob 

      # self.T.append(self.T_raw[(r,c)])

class TestDirections(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    self.grid = Grid(w=4,h=3, default_reward=-0.04, default_bad_reward=-1)
    self.grid.add_invalid((1,1))
    self.grid.add_reward(p=(0,3), amt=1)
    self.grid.add_reward(p=(1,3), amt=-1)
    self.grid.add_terminal((0,3))
    self.grid.add_terminal((1,3))
    
    self.d = Directions(grid=self.grid)
    self.d.set_probabilities(f=0.8,s=0.1,b=0)
    
    
    self.mdp = GridMDP(w=4,
                      h=3,
                      motion_forward_prob=0.8,
                      motion_side_prob=0.1,
                      motion_back_prob=0,
                      grid=self.grid)
    
    self.mdp.grid.add_invalid((1,1))
    self.mdp.compute_state_transition_matrix()
    
    super(TestDirections, self).__init__(*args, **kwargs)
  def test_directions(self):
    self.assertEqual(self.d.get_coordinate_from_direction(p=(10,5), direction=0), (9,5))
    self.assertEqual(self.d.get_coordinate_from_direction(p=(10,5), direction=1), (10,6))   
    
    
    results = self.d.get_target_coordinate_and_prob_from_direction((0,0), 0)
    self.assertEqual(sorted(results), sorted([((-1, 0), 0.8), ((0, -1), 0.1), ((0, 1), 0.1)]))
    results = self.d.get_resulting_coordinate_and_prob_from_direction((0,0), 0)
    self.assertEqual(sorted(results), sorted([((0, 0), 0.8), ((0, 0), 0.1), ((0, 1), 0.1)]))
     

  def test_grid_label(self):
    self.assertEqual(self.grid.to_p(10), (2,2))
    self.assertEqual(self.grid.to_s((2,2)), 10)
  
  def test_t(self):
    p = self.mdp.grid.to_p(8)
    self.assertEqual(self.mdp.P[p][2][self.mdp.grid.to_p(8)], 0.9)
    self.assertEqual(self.mdp.P[p][2][self.mdp.grid.to_p(9)], 0.1)
    #test bad boundary, should hit and go back so P(1') == .8
    self.assertEqual(self.mdp.P[self.mdp.grid.to_p(1)][2][self.mdp.grid.to_p(1)], 0.8)
  
  def test_policy_iteration(self):
    policy = {
      (0,0):2,
      (0,1):2,
      (0,2):3,
      (0,3):None,
      (1,0):0,
      (1,1):None,
      (1,2):2,
      (1,3):None,
      (2,0):1,
      (2,1):2,
      (2,2):1,
      (2,3):0
    }
    self.mdp.build_policy(strategy = PolicyIterationStrategy, starting_policy=policy)
    
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
  print(mdp.policy)
  print(mdp.get_action(s=0))


