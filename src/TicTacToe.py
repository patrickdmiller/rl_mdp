'''TODO:
make winning states have winning point value vs going to 'win' state. just give rewards high values for winning states. 
'''

import numpy as np
from hashlib import sha1
import unittest
import pickle as pk
import os
from abc import ABC, abstractmethod
from collections import defaultdict
class TicTacToeState:
  @classmethod
  def create(cls, state_key, n, rot=0, win_counter=None):
    l = np.array(cls.key_to_list(state_key))
    l = np.rot90(l.reshape([n,n]), k = -rot)
    new_state = TicTacToeState(n)
    for r in range(n):
      for c in range(n):
        if l[r,c]!=0:
          is_win = new_state.do_move(r,c, l[r,c])
          if is_win and win_counter:
            win_counter[l[r,c]]+=1
    return new_state

  @classmethod
  def key_to_list(cls, state_key):
    l = []
    i = 0
    while i < len(state_key):
      if state_key[i] == '0':
        l.append(0)
      elif state_key[i] == '1':
        l.append(1)
      else:
        l.append(-1)
        i+=1
      i+=1
    return l
  
  def __init__(self, n):
    self.n = n
    self.m = np.zeros([n,n], dtype=np.int8)
    self.rows = [0 for i in range(n)]
    self.cols = [0 for i in range(n)]
    self.diag_d = 0
    self.diag_u = 0
    self.state_key_cached = None
    self.rotation = 0
  
  def hash(self, a):
    return ''.join(list(map(lambda c:''.join(list(map(lambda x: str(int(x)), c))), a)))
  
  def key_position_to_r_c(self,pos):
    r = pos // self.n
    c = pos - (r * self.n)
    return (r,c)
  def rotate_r_c_right(self,r,c,rot):
    for i in range(rot):
      rp,cp = r,c
      r = cp
      c = self.n-rp-1
      # print("rotation", i, ">", r,c)
    return r,c
  
  def move_for_next_state_key(self, next_state_key):
    
    # print("need to find origin for ", next_state_key)
    next_state_key_l = self.key_to_list(next_state_key)
    
    all_keys = self.state_key(with_rotations = True)
    key_for_next_state_key, move_r, move_c, move_rotation = None, None, None, None
    for key in all_keys['keys']:
      key_l = self.key_to_list(key)
      count = 0
      move = None
      for i,c in enumerate(key_l):
        if c != next_state_key_l[i]:
          count+=1
          move = i
          if count > 1:
            break
      if count == 1:
        # print("found key: ", key, "rotation: ", all_keys['keys'][key], "currently at rotation: ", self.rotation, "move", move)
        key_for_next_state_key = key
        move_r,move_c = self.key_position_to_r_c(move)
        move_rotation = all_keys['keys'][key]
        break
    
    # print("done", key_for_next_state_key)
    # print("rc in string is ", move_r, move_c, "adjusting for rotation of ", move_rotation)
    move_r, move_c = self.rotate_r_c_right(move_r, move_c, move_rotation)
    # print("we need to do move", move_r, move_c)
    return (move_r, move_c)
    
        
  def do_move(self, r,c, piece):
    # print("doing move", r,c,piece)
    self.m[r,c] = piece
    self.state_key(skip_cache = True)
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
    self.state_key(skip_cache = True)
    self.rows[r]+=piece
    self.cols[c]+=piece
    if r == c: 
      self.diag_d+=piece
    if r + c == self.n-1:
      self.diag_u+=piece
    
    
  def state_key(self, with_rotations=False, skip_cache=False):
    if not skip_cache and not with_rotations and self.state_key_cached is not None:
      return self.state_key_cached
    
    _m = self.m
    hashes = [self.hash(_m)]
    key_rotation_map = {self.hash(_m):0}
    for i in range(3):
      _m = np.rot90(_m)
      hashes.append(self.hash(_m))
      key_rotation_map[self.hash(_m)] = i+1
    

    hashes.sort()
    self.rotation = key_rotation_map[hashes[0]]
    if with_rotations:
      return {'keys':key_rotation_map, 'key':hashes[0], 'rotation_for_key':key_rotation_map[hashes[0]]}
    self.state_key_cached = hashes[0]
    return hashes[0]
  
  def find_state_and_rotation(self, state_map):
    if self.state_key() not in state_map:
      raise Exception(" ( O _O) State key not in state_map. Problematic key:", self.state_key())
    keys = self.state_key(with_rotations=True)
    
    return {'state_key':self.state_key(), 'next_states' : state_map[self.state_key()], 'rotation':keys[self.state_key()]}
  def __repr__(self):
    return str(self.m)
  
  def display_state(self, with_rotation=True):
    pass

class TicTacToeStateMap:
  @classmethod
  def load_or_init(cls, n, force_init = False, *args, **kwargs):
    if not force_init and os.path.isfile(os.path.join('./', f'ttt_{n}_pickle.p')):
      print("loading from file")
      with open(os.path.join('./', f'ttt_{n}_pickle.p'), 'rb') as pickle_file:
        return pk.load(pickle_file)
    return cls(n=n, *args, **kwargs)
  
  def __init__(self, n, default_reward=0, default_win_reward=100, default_lose_reward=-100, default_tie_reward = 10):
    self.n = n
    self._empty = TicTacToeState(n = self.n)
    self.state_map = {self._empty.state_key():set()}
    self.empty_state = self.state_map[self._empty.state_key()]
    self.rewards = {-1:{}, 1:{}}
    
    self.default_reward = default_reward
    self.default_win_reward=default_win_reward
    self.default_lose_reward=default_lose_reward
    self.default_tie_reward = default_tie_reward
    self.build_state_map()
    self.pickle()
    
  def build_state_map(self):
    state = TicTacToeState(n = self.n)

    def build(state, piece, from_state, count):
      for r in range(self.n):
        for c in range(self.n):
          if state.m[r,c] == 0:
            
            #if this state is already in the from state it's already been processed. 
            if state.state_key() in from_state:
              return
            
            is_win = state.do_move(r,c,piece)
            count+=1
            if is_win:
              #set win rewards
              self.rewards[piece][state.state_key()] = self.default_win_reward
              self.rewards[piece*-1][state.state_key()] = self.default_lose_reward
            
            #add it for backtrack forward
            from_state.add(state.state_key())
            if state.state_key() not in self.state_map:
              if not is_win and count == self.n**2:
                #tie game
                self.rewards[piece][state.state_key()] = self.default_tie_reward
                self.rewards[piece*-1][state.state_key()] = self.default_tie_reward
              self.state_map[state.state_key()] = set()
              # if from_state is not None:
              if not is_win:
                build(state=state, piece=piece*-1, from_state = self.state_map[state.state_key()], count = count)
           
            #remove it for backtrack back
            state.clear_move(r,c)
            count-=1
    build(state,1, self.empty_state, 0)
  def get_reward(self, state_key, piece):
    if state_key in self.rewards[piece]:
      return self.rewards[piece][state_key]
    return self.default_reward
  def get_neighbors(self, state_key):
    return self.state_map[state_key]
  def pickle(self):
    with  open(os.path.join('./', f'ttt_{self.n}_pickle.p'), 'wb') as pickle_file:
      pk.dump(self,pickle_file)
  #act like a dictionary if called by key
  def __getitem__(self, key):
    return self.state_map[key]
  def keys(self):
    return self.state_map.keys()
  def is_terminal(self, state_key):
    return len(self[state_key]) == 0
class TicTacToe:
  def __init__(self, n, force_init = False):
    self.n = n
    self.state_map = TicTacToeStateMap.load_or_init(n, force_init = force_init)
    self.current_state = self.state_map._empty
    self.turn = None
    self.state = TicTacToeState(n = self.n)
  def reset_state(self):
    self.state = TicTacToeState(n = self.n)
  def train_agent(self, trainee, opponent, max_attempts_for_change = 100, max_learning_epochs = 100000):
    # max_learning_epochs = 10000
    games_results =[]
    for i in range(max_learning_epochs):
      if i % 100 == 0:
        print(i, end="..")
        if i % 1000 == 0:
          print("")
      if trainee.last_change > max_attempts_for_change:
        print(f'no change in at least {max_attempts_for_change}')
        break
      player1 = trainee
      player2 = opponent
      result = self.start_game(player1 = player1, player2 = player2)
      player1.clear_memory()
      if result[0] == 'TIE':
        games_results.append('TIE')
      else:
        if result[1] == 1:
          games_results.append('WIN')
        else:
          games_results.append('LOSE')
      # self.reset_state()
      player2= trainee
      player1= opponent
      result = self.start_game(player1 = player1, player2 = player2)
     
      player2.clear_memory()
      if result[0] == 'TIE':
        games_results.append('TIE')
      else:
        if result[1] == -1:
          games_results.append('WIN')
        else:
          games_results.append('LOSE')
      # self.reset_state()
    print("done training.")
    print(games_results)
      
    
  def start_game(self, player1, player2, debug = False):
    self.reset_state()
    self.player1 = player1
    self.player1.init_game(state_map = self.state_map, piece = 1, debug = debug)
    self.player2 = player2
    self.player2.init_game(state_map = self.state_map, piece = -1, debug = debug)
    self.turn = self.player1
    self.state = TicTacToeState(n = self.n)
    opening = True
    max_turns = self.n * self.n
    turn_count = 0
    outcome = ('TIE',0)
    while True and turn_count < max_turns:
      
      turn_count+=1
      move = self.turn.process_state(self.state, reward=self.state_map.get_reward(self.state.state_key(), piece=self.turn.piece))
      if debug:
        print("move: ", move)
      did_win = False
      if 'move' in move:
        did_win = self.state.do_move(move['move'][0][0], move['move'][0][1], self.turn.piece)
      elif 'state' in move:
        #firgure out the move needed to get there
        r, c = self.state.move_for_next_state_key(next_state_key=move['state'])
        did_win = self.state.do_move(r,c,self.turn.piece)
      else:
        raise Exception("did not receive valid output from process_state")
      if opening and self.turn.t == 'agent':
        if debug:
          print("first move by agent. we will rotate")
        rot = np.random.randint(4)
        self.state.rotation = rot
      if opening:
        opening = False
      if debug:
        print("the current board state is: ", self.state.state_key(), self.state.rotation)
        print(self.state)
      if did_win:
        if debug:
          print("WINNER")
        outcome = ('WIN',self.turn.piece)
        break
      
      if self.turn == self.player1:
        self.turn = player2
      else:
        self.turn = player1
      #let the agents know the outcome
    player1.process_state(self.state, reward=self.state_map.get_reward(self.state.state_key(), piece=player1.piece))
    player2.process_state(self.state, reward=self.state_map.get_reward(self.state.state_key(), piece=player2.piece))
    if debug:
      print("DONE")
    player1.finish()
    player2.finish()
    return outcome
    
    
class TicTacToeStrategy(ABC):
  def __init__(self, t='agent', debug = False, **kwargs):
    self.t = 'agent'
    self.debug = debug
    self.build()
  @abstractmethod
  def process_state(self, state, reward=0):
    pass
  
  @abstractmethod
  def build(self,):
    pass
  def finish(self):
    pass

  #assumes subclass populated state_map etc..
  def build_policy(self, piece):
    did_change =  False
    for key in self.policy[piece]:
      best_utility = -float('inf')
      best_neighbors = []
      
      for neighbor in self.state_map.get_neighbors(state_key = key):
        if self.utilities[piece][neighbor] >= best_utility:
          if self.utilities[piece][neighbor] > best_utility:
            #just add it
            best_neighbors = [neighbor]
          else:
            best_neighbors.append(neighbor)
          best_utility = self.utilities[piece][neighbor]
          
      #if current neighbor is not in the possible best, then we changed, otherwise, juts leave it the same
      if best_neighbors and self.policy[piece][key] not in best_neighbors:
        # print(key)
        # print(best_neighbors)
        did_change = True
        self.policy[piece][key] = np.random.choice(best_neighbors)
    return did_change
      
  def init_game(self, state_map, piece, debug=False, **kwargs):
    self.state_map = state_map
    self.piece = piece
    self.debug = debug
    if self.debug:
      print("debug mode on")
      
  def process_state(self, state, reward=0):
    if state.state_key() not in self.policy[self.piece]:
      if self.state_map.is_terminal(state.state_key()):
        return
        raise Warning("key not in policy. likely terminal")
    if self.debug:
      print("received state", state.state_key())
      print("policy: ", self.policy[self.piece][state.state_key()])
      print("values from this state: ")
      for key in self.state_map[state.state_key()]:
        print("\t", key, " : ", self.utilities[self.piece][key])
    return {'state': self.policy[self.piece][state.state_key()]}

class TestStrategy(TicTacToeStrategy):
  def __init__(self, t):
    self.moves = []
    self.i = 0
    self.t = t
  
  def build(self):
    pass
  #move should be either {'state':_STATE_KEY} or {'move':((r,c),piece)}
  def process_state(self, state, reward=0):
    move = None
    if self.t == 'human':
      move = {'move':(self.moves[self.i], self.piece)}
      
    if self.t == 'agent':
      print('valid moves. ', self.state_map.get_neighbors(state.state_key()))
      move = {'state':self.moves[self.i]}
      
    self.i+=1
    return move

class PolicyIterationStrategy(TicTacToeStrategy):
  def __init__(self, state_map, gamma = 0.5, delta_convergence_threshold=1, default_reward=0,*args, **kwargs):
    
    self.gamma = gamma
    self.state_map = state_map
    self.policy = {-1:{}, 1:{}}
    self.utilities = {-1:{}, 1:{}}
    self.delta_convergence_threshold=delta_convergence_threshold
    super().__init__(t='agent',*args, **kwargs)

  def build(self, **kwargs):
    #build policy and utilities as empty graphs
    for key in self.state_map.keys():
      if len(self.state_map[key]) > 0:
        self.policy[-1][key] = np.random.choice(list(self.state_map[key]))
        self.policy[1][key] = np.random.choice(list(self.state_map[key]))
      self.utilities[-1][key] = 0
      self.utilities[1][key] = 0
    
    #evaluate and build_policy
    for piece in [-1, 1]:
      max_iterations = 100
      loops = 0
      while True and loops < max_iterations:
        loops+=1
        self.evaluate(piece=piece)
        if not self.build_policy(piece = piece):
          if self.debug:
            print("no changes")
          break
        if self.debug:
          print("changes. looping")
    if self.debug:
      print(self.utilities)
  
  def evaluate(self, piece = 0):
    utility = self.utilities[piece]
    for i in range(50):
      delta = 0
      for key in self.state_map.keys():
        u = self.state_map.get_reward(state_key=key, piece= piece)
        if self.state_map.is_terminal(state_key=key):
          #this is a terminal state. just give
          u += ( self.gamma * utility[key] )
        else:
          policy_points_to = self.policy[piece][key]
          u+= (self.gamma * self.utilities[piece][policy_points_to])
        delta += abs(utility[key] - u)
        utility[key] = u
      if self.debug:
        print(delta)
      if delta <= self.delta_convergence_threshold:
        break
    # print(utility)
    
class ValueIterationStrategy(TicTacToeStrategy):
  def __init__(self, state_map, gamma = 0.5, delta_convergence_threshold=1, default_reward=0, **kwargs):
    self.policy = {-1:{}, 1:{}}
    self.gamma = gamma
    self.state_map = state_map
    self.utilities = {-1:{}, 1:{}}
    self.delta_convergence_threshold=delta_convergence_threshold

    super().__init__(t='agent', **kwargs)
  
  def build(self, **kwargs):
    for key in self.state_map.keys():
      if len(self.state_map[key]) > 0:
        self.policy[-1][key] = -1
        self.policy[1][key] = -1
      self.utilities[-1][key] = 0
      self.utilities[1][key] = 0

    #evaluate and build_policy
    for piece in [-1,1]:
      self.iterate(piece=piece)
      self.build_policy(piece = piece)
      if self.debug:
        print("changes. looping")

    
  def iterate(self, piece = 0):
    utility = self.utilities[piece]
    for i in range(50): #max iterations
      delta = 0
      for key in self.state_map.keys():
        u = self.state_map.get_reward(state_key=key, piece= piece)
        if self.state_map.is_terminal(state_key=key):
          u += ( self.gamma * utility[key] )
        else:
          for next_key in self.state_map[key]:
            u+=( self.gamma * utility[next_key])
        delta += abs(utility[key] - u)
        utility[key] = u
      if self.debug:
        print(delta)
      if delta <= self.delta_convergence_threshold:
        break


class QLearningStrategy(TicTacToeStrategy):
  def __init__(self, state_map, gamma = 0.5, delta_convergence_threshold=1, default_reward=0, **kwargs):
    #state_map is a handy way to know next valid moves. we are not using it to make the policy
    self.memory = None #(state_key, action_state_key) #from state key we went to...
    self.gamma = gamma
    self.state_map = state_map
    self.Q = {-1:{}, 1:{}}
    self.alpha = 0.5
    self.gamma = gamma
    self.last_change = 0
    super().__init__(t='agent', **kwargs)
  def finish(self):
    self.clear_memory()
  def build(self, **kwargs):
    #build Q table
    for key in self.state_map.keys():
      self.Q[1][key] = {} #list of all actions... which are keys it can move to
      self.Q[-1][key] = {}
      for next_key in self.state_map[key]:
        self.Q[-1][key][next_key] = 0
        self.Q[1][key][next_key] = 0
    
  def clear_memory(self):
    self.memory = None
    
  def get_max_action_value_from_state_key(self, state_key, piece):
    if self.state_map.is_terminal(state_key):
      return self.state_map.get_reward(state_key, piece)
    
    max_action_value = -float('inf')
    for next_key in self.Q[piece][state_key]:
      max_action_value = max( max_action_value, self.Q[piece][state_key][next_key])
    return max_action_value
  def get_max_actions_from_state_key(self, state_key, piece):
    if self.state_map.is_terminal(state_key):
      return [state_key]
    max_action_value = -float('inf')
    max_actions = []
    for next_key in self.Q[piece][state_key]:
      if self.Q[piece][state_key][next_key] >= max_action_value:
        if self.Q[piece][state_key][next_key] == max_action_value:
          max_actions.append(next_key)
        else:
          max_action_value = self.Q[piece][state_key][next_key]
          max_actions = [next_key]
    return max_actions
  def process_state(self, state,  reward, learning=True, piece = None): #reward is th reward you got for going INTO this spot. so you need to remember your last move.
    if self.debug:
      print("received ", reward, state.state_key())
    if piece == None:
      piece = self.piece
    Q = self.Q[piece]
    
    if self.memory != None and learning:
      # reward = self.state_map.get_reward(state_key = state.state_key(), piece=piece)
      previous_Q = Q[self.memory[0]][self.memory[1]]
      Q[self.memory[0]][self.memory[1]] = ((1-self.alpha) * Q[self.memory[0]][self.memory[1]] ) + self.alpha * ( reward + (self.gamma * self.get_max_action_value_from_state_key(state_key=state.state_key(), piece=piece)))  
      if previous_Q !=  Q[self.memory[0]][self.memory[1]]:
        self.last_change+=1
      else:
        self.last_change = 0
      if self.debug:
        print("Q is updated: ", Q[self.memory[0]][self.memory[1]])
      
    #pick the next move
    
    action = np.random.choice(self.get_max_actions_from_state_key(state.state_key(), piece))
    self.memory = (state.state_key(), action)
    return {'state': action}
class RandomStrategy(TicTacToeStrategy):
  
  def process_state(self, state, reward=0):
    # print("here")
    if self.state_map.is_terminal(state.state_key()):
      return None
    action = np.random.choice(list(self.state_map[state.state_key()]))
    return {'state':action}
  
  def build(self):
    pass
  
class HumanStrategy(TicTacToeStrategy):
  def __init__(self, *args, **kwargs):
    super().__init__(t='human', *args, **kwargs)

  def build(self):
    pass

  def process_state(self, state, reward=0):
    if self.state_map.is_terminal(state.state_key()):
      return None
    invalid = True
    while invalid:
      
      in_coord = input()
      if in_coord=='':
        continue
      r,c = in_coord.split(',')
      r = int(r)
      c = int(c)
      if state.m[r,c] == 0:
        invalid = False
    return {'move':((r,c), self.piece)}
    print(in_coord)
    
if __name__ == '__main__':
  class Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
      super(Test, self).__init__(*args, **kwargs)
      
    def test_ttt(self):
      t = TicTacToe(3, force_init = True) 
      player1_strategy = TestStrategy(t='human')
      player2_strategy = TestStrategy(t='agent')
      player1_strategy.moves.append((0,2))
      player2_strategy.moves.append('000-100100')
      player1_strategy.moves.append((1,1))
      player2_strategy.moves.append('0-1101-1000')
      player1_strategy.moves.append((2,0))
      
      t.start_game(player1 = player1_strategy, player2 = player2_strategy)
      
      self.assertEqual(t.state.state_key(), '0-1101-1100')
      
      #check the reward
      self.assertEqual(t.state_map.get_reward(state_key='0-1101-1100', piece=1), t.state_map.default_win_reward)
      self.assertEqual(t.state_map.get_reward(state_key='0-1101-1100', piece=-1), t.state_map.default_lose_reward)
      self.assertEqual(t.state_map.get_reward(state_key='000000001', piece=1), t.state_map.default_reward)
      self.assertEqual(t.state_map.get_reward(state_key='000000001', piece=-1), t.state_map.default_reward)
      self.assertEqual(t.state_map.get_reward(state_key='-11-1-1111-11', piece=1), t.state_map.default_tie_reward)
      self.assertEqual(t.state_map.get_reward(state_key='-11-1-1111-11', piece=-1), t.state_map.default_tie_reward)
      self.assertEqual(set(t.state_map['000000001']), set(('000-100100', '0-10000100', '-100000001', '-100000100', '0-11000000', '0-10000001', '-101000000', '0000-10001',)))
      self.assertTrue(t.state_map.is_terminal(state_key='-11-1-1111-11'))
      self.assertFalse(t.state_map.is_terminal(state_key='000000000'))
  # unittest.main()
  # 3! = 362880

  t = TicTacToe(4)
  print("loaded")
  # player1 = PolicyIterationStrategy(state_map = t.state_map)
  # player2 = ValueIterationStrategy(state_map = t.state_map)
  player2 = QLearningStrategy(state_map = t.state_map)
  print("training")
  # t.train_agent(trainee=player2, opponent_strategy=RandomStrategy, max_attempts_for_change=1000)
  t.train_agent(trainee=player2, opponent=RandomStrategy(t.state_map), max_attempts_for_change=1000, max_learning_epochs = 300000)
  
  # print("training...")
  t.train_agent(trainee=player2, opponent=PolicyIterationStrategy(t.state_map), max_attempts_for_change=10000, max_learning_epochs=10000)
  # t.train_agent(trainee=player2, opponent=ValueIterationStrategy(t.state_map), max_attempts_for_change=1000)
  # player1 = PolicyIterationStrategy(state_map = t.state_map)
  player1 = HumanStrategy()
  #HumanStrategy()
  while True:
    print("new game")
    t.start_game(player1 = player1, player2 = player2, debug=True)
    
  # for key in t.state_map.keys():
  #   for next_key in player2.Q[-1][key]:
  #     if player2.Q[-1][key][next_key] != 0:
  #       print(player2.Q[-1][key][next_key])

  # print(player2.Q[-1])
  # print(player2.Q[-1])
  #all winning states
  # print("Winners")
  # win_counter = {-1:0, 1:0}
  # for key in t.state_map:
  #   if 'win1' in t.state_map[key]:
  #     print(key)
  #     TicTacToeState.create(key, 3, win_counter=win_counter)
  # print(win_counter)