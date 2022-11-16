'''
author: patrick miller
gihub: github.com/patrickdmiller
gatech: pmiller75
'''

import unittest
import numpy as np
from abc import ABC, abstractmethod
from utilities import Utility, Directions

class PolicyStrategy:
  def __init__(self):
    pass
  
  @abstractmethod
  def build(self):
    pass

  @abstractmethod
  def get_policy_for_state(self, p):
    pass
  
  def pp(self, which=None):
    if not which:
      which = self.policy
    if not self.grid or not self.policy:
      return
    print("size: ", self.grid.w, 'x', self.grid.h)
    for r in range(self.grid.h):
      print("")
      for c in range(self.grid.w):
        print(which[(r,c)], sep=" | ", end="\t|")
    print("\n----------")

class ValueIterationStrategy(PolicyStrategy):
  def __init__(self, grid, dirs, gamma=0.5): #inject grid and policy
    self.grid = grid
    self.policy = None
    self.dirs = dirs
    self.gamma = gamma
    self.history = []
    self.utilities = Utility(w = self.grid.w, h=self.grid.h)
    
  def build(self, starting_policy = None):
    #if a starting policy , use it
    if starting_policy:
      self.policy = starting_policy
    else:
      #random
      self.policy = self.grid.empty_states(0)
        #generate a random policy
      for r in range(self.grid.h):
        for c in range(self.grid.w):
          if self.grid.is_valid((r,c)) and not self.grid.is_bad((r,c)):
            self.policy[(r,c)] = np.random.randint(0, len(self.dirs))
    self.pp(which=self.utilities)
    self.iterate()
    # max_iterations = 500
    # loops = 0
    # while True and loops < max_iterations:
    #   loops+=1
    #   self.evaluate()
    #   if not self.improve():
    #     break
    self.pp(which=self.utilities)
  
  def iterate(self):
    #iterate over each space
    for r in range(self.grid.h):
      for c in range(self.grid.w):
        #get the best utility from moving in all directions
        if self.grid.is_valid((r,c)):
          new_u = self.grid.get_reward((r,c))
          print(new_u)
          if not self.grid.is_terminal((r,c)):
            #try in each direction to find the max utility FROM this spot, then add the default
            
            best_utility = -float('inf')
            best_direction = -1
                    
            for direction in range(len(self.dirs)):
              next_positions = self.dirs.get_resulting_coordinate_and_prob_from_direction(p=(r,c), direction=direction)
              u = 0
              for np, prob in next_positions:
                u += (self.gamma * self.utilities[(np)] * prob)
              if u > best_utility:
                best_utility = u
                best_direction = direction
            new_u += best_utility
            
          else:
          #if it's termina. itself. 
            new_u += (self.gamma * self.utilities.get((r,c)))

          self.utilities.set((r,c), new_u)
      
  def get_policy_for_state(self, p):
    return self.policy[p]

  def get_utility_for_state(self, p):
    return self.utilities.get(p)
class PolicyIterationStrategy(PolicyStrategy):
  def __init__(self, grid, dirs, gamma=0.5): #inject grid and policy
    self.grid = grid
    self.policy = None
    self.dirs = dirs
    self.gamma = gamma
    self.history = []
    self.utilities = Utility(w = self.grid.w, h=self.grid.h)
    
  def build(self, starting_policy = None):
    #if a starting policy , use it
    if starting_policy:
      self.policy = starting_policy
    else:
      #random
      self.policy = self.grid.empty_states(0)
        #generate a random policy
      for r in range(self.grid.h):
        for c in range(self.grid.w):
          if self.grid.is_valid((r,c)) and not self.grid.is_bad((r,c)):
            self.policy[(r,c)] = np.random.randint(0, len(self.dirs))
    
    max_iterations = 500
    loops = 0
    while True and loops < max_iterations:
      loops+=1
      self.evaluate()
      if not self.improve():
        break
    self.pp()
  
  def get_policy_for_state(self, p):
    return self.policy[p]

  def get_utility_for_state(self, p):
    return self.utilities.get(p)
 
  def evaluate(self):
    #iterate over each 
    for i in range(50):
      delta = 0
      for r in range(self.grid.h):
        for c in range(self.grid.w):
          u = self.grid.get_reward((r,c))
          debug_s = f'v{self.grid.to_s((r,c))} = {u}'
          if not self.grid.is_terminal((r,c)) and self.grid.is_valid((r,c)):
            results = self.dirs.get_resulting_coordinate_and_prob_from_direction(p=(r,c), direction=self.policy[(r,c)])
            for p, prob in results:
              debug_s+=f'+({self.gamma} * {prob} * {self.utilities.get(p)})'
              u += (self.gamma * prob * self.utilities.get(p))
          else:
              u += (self.gamma * self.utilities.get((r,c)))
          delta += abs(self.utilities.get((r,c)) - u)
          
          self.utilities.set((r,c), u)
          
          debug_s+=f' = {u}'
          # print(debug_s)

        
      self.history.append(delta)
      # print(self.utilities)
      # print(self.policy)
  def improve(self):
    did_change = False
    for r in range(self.grid.h):
      for c in range(self.grid.w):
        if self.grid.is_valid((r,c)) and not self.grid.is_terminal((r,c)):
          best_direction = -1
          best_utility = -float('inf')
          # for p, direction in self.dirs.get_valid_neighbors_and_directions(p=(r,c)):
          
          for direction in range(len(self.dirs)):
          #which direction has the highest utility?
            next_positions = self.dirs.get_resulting_coordinate_and_prob_from_direction((r,c), direction)
            u = 0
            for np, prob in next_positions:
              u += (self.utilities.get(np) * prob)
            # if r == 7 and c == 6:
            #   print(direction, u)
            if u > best_utility:
              best_utility = u
              best_direction = direction
          # if r == 7 and c == 6:
          #   print("best = ", best_direction)
          if best_direction > -1:
            if self.policy[(r,c)] != best_direction:
              self.policy[(r,c)] = best_direction
              did_change = True

          
    return did_change

class Grid:
  def __init__(self,w,h, default_reward=0, default_bad_reward=-1):
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
    return self.default_reward  
  
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
      raise Exception("No strategy defined")
    else:
      self.policy = starting_policy
      self.strategy = strategy(grid = self.grid, dirs = self.dirs)
      self.strategy.build()

  def get_action(self, p = None, s = None):
    if p is None:
      p = self.grid.to_p(s)
    return self.strategy.get_policy_for_state(p)


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
    
    self.assertEqual(self.d.get_coordinate_from_direction((1,2), 1), (1,3))
    
    self.assertEqual(sorted(self.d.get_valid_neighbors_and_directions((1,2))), sorted([((0,2), 0), ((2,2), 2), ((1,3), 1)])) #(1,1) is invalid

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
    
    self.mdp.build_policy(strategy=ValueIterationStrategy, starting_policy = policy)
    
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


