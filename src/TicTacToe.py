import numpy as np
from hashlib import sha1
import unittest
import pickle as pk
import os
from abc import ABC, abstractmethod
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
        print("found key: ", key, "rotation: ", all_keys['keys'][key], "currently at rotation: ", self.rotation, "move", move)
        key_for_next_state_key = key
        move_r,move_c = self.key_position_to_r_c(move)
        move_rotation = all_keys['keys'][key]
        break
    
    print("done", key_for_next_state_key)
    print("rc in string is ", move_r, move_c, "adjusting for rotation of ", move_rotation)
    move_r, move_c = self.rotate_r_c_right(move_r, move_c, move_rotation)
    print("we need to do move", move_r, move_c)
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
  
class TicTacToe:
  def __init__(self, n):
    self.n = n
    self._empty = TicTacToeState(n = self.n)
    self.state_map = {self._empty.state_key():set(), 'win1':set(), 'win-1':set()}
    self.empty_state = self.state_map[self._empty.state_key()]
    self.first = None
    self.current_state = self._empty
    self.turn = None
    self.state = TicTacToeState(n = self.n)
    print("init")
  def start_game(self, player1, player2):
    #first player is always x
    self.player1 = player1
    self.player1.init_game(state_map = self.state_map, piece = 1)
    self.player2 = player2
    self.player2.init_game(state_map = self.state_map, piece = -1)
    self.turn = self.player1
    self.state = TicTacToeState(n = self.n)
    opening = True
    max_turns = self.n * self.n
    turn_count = 0
    while True and turn_count < max_turns:
      turn_count+=1
      move = self.turn.process_state(self.state)
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
        print("first move by agent. we will rotate")
        rot = np.random.randint(4)
      if opening:
        opening = False
      print("the current board state is: ", self.state.state_key(), self.state.rotation)
      print(self.state)
      if did_win:
        print("WINNER")
        break
      if self.turn == self.player1:
        self.turn = player2
      else:
        self.turn = player1
    print("DONE")
      # return
        #we select a random rotation for the opening move since any move is valid at this point and the agents know nothing of rotation. this keeps opening moves fresh.
  def build_state_map(self, force=False):
    if not force and os.path.isfile(os.path.join('./', f'ttt_{self.n}_pickle.p')):
      print("ttt state map file found")
      with open(os.path.join('./', f'ttt_{self.n}_pickle.p'), 'rb') as pickle_file:
        self.state_map = pk.load(pickle_file)
        return
    else:
      print("ttt not found. recreating")
    state = TicTacToeState(n = self.n)
    debug = set()
    def build(state, piece, from_state):
      # print("piece is ", piece)
      # print(state)
      for r in range(self.n):
        for c in range(self.n):
          if state.m[r,c] == 0:
            #add it
            # state.m[r,c] = piece
            is_win = state.do_move(r,c,piece)
            # if is_win:
            #   # print("winner: ", state.state_key(), state)
            #   if 
            #   from_state.add('win')
            # else:
            if state.state_key() in from_state:
              return
            from_state.add(state.state_key())
            
            if state.state_key() not in self.state_map:
              self.state_map[state.state_key()] = set()
              # if from_state is not None:
              if not is_win:
                build(state=state, piece=piece*-1, from_state = self.state_map[state.state_key()])
              #remove it
            if is_win:

              self.state_map[state.state_key()].add(f'win{piece}')
            state.clear_move(r,c)
                # break
    build(state,1, self.empty_state)
    # print(self.state_map)
    #save it to file
    with  open(os.path.join('./', f'ttt_{self.n}_pickle.p'), 'wb') as pickle_file:
      pk.dump(self.state_map,pickle_file)
    
  def get_state_for_player(self, player):
    if player != 0 and player != 1:
      raise Exception("player should be 0 or 1")

    
  
class TicTacToeStrategy(ABC):
  def __init__(self, t='agent', **kwargs):
    self.t = 'agent'
  
  @abstractmethod
  def process_state(self, state, reward=0):
    pass
  
  @abstractmethod
  def build(self,):
    pass
  
  def init_game(self, state_map, piece):
    self.state_map = state_map
    self.piece = piece

#just for unit tests. runs through moves in order. 
class TestStrategy(TicTacToeStrategy):
  def __init__(self, t):
    self.moves = []
    self.i = 0
    self.t = t
  
  def build(self):
    pass
  #move should be either {'state':_STATE_KEY} or {'move':((r,c),piece)}
  def process_state(self, state, reward = 0):
    move = None
    if self.t == 'human':
      move = {'move':(self.moves[self.i], self.piece)}
      
    if self.t == 'agent':
      print('valid moves. ', self.state_map[state.state_key()])
      move = {'state':self.moves[self.i]}
      
    self.i+=1
    return move

class PolicyIterationStrategy(TicTacToeStrategy):
  def __init__(self, state_map, gamma = 0.5, delta_convergence_threshold=1, default_reward=0):
    self.policy = {-1:{}, 1:{}}
    self.gamma = gamma
    self.state_map = state_map
    self.utilities = {-1:{}, 1:{}}
    self.rewards = {-1:{}, 1:{}}
    self.delta_convergence_threshold=delta_convergence_threshold
    self.default_reward=default_reward
    super().__init__(t='agent')
    
  def process_state(self, state, reward=0):
    return {'state': self.policy[self.piece][state.state_key()]}

  def build(self, **kwargs):
    #build policy and utilities as empty graphs
    for key in self.state_map:
      if len(self.state_map[key]) > 0:
        self.policy[-1][key] = np.random.choice(list(self.state_map[key]))
        self.policy[1][key] = np.random.choice(list(self.state_map[key]))
      else:
        self.policy[-1][key] = None
        self.policy[1][key] = None
        if key != 'win1' and key != 'win-1':
          #cats game
          self.rewards[-1][key] = -20
          self.rewards[1][key] = -20
        
        
      self.utilities[-1][key] = 0
      self.utilities[1][key] = 0
      
    self.rewards[-1]['win-1'] = 100
    self.rewards[-1]['win1'] = -200
    self.rewards[1]['win1'] = 100
    self.rewards[1]['win-1'] = -200
    
    #evaluate and build_policy
    for piece in [1]:
      max_iterations = 100
      loops = 0
      while True and loops < max_iterations:
        loops+=1
        self.evaluate(piece=piece)
        if not self.build_policy(piece = piece):
          print("no changes")
          break
        print("changes. looping")
    
  
  def evaluate(self, piece = 0):
    utility = self.utilities[piece]
    for i in range(50):
      delta = 0
      for key in self.state_map:
        
        u = 0
        if key in self.rewards[piece]:
          u = self.rewards[piece][key]
        else:
          u = self.default_reward
        if key == 'win-1' or key == 'win1' or self.policy[piece][key]==None : #win or cats game, just take the reward
          #this is a terminal state. just give
          u += ( self.gamma * utility[key] )
        else:
          policy_points_to = self.policy[piece][key]
          
          u+= (self.gamma * self.utilities[piece][policy_points_to])
          # for next_key in self.state_map[key]:
          #   u+=( self.gamma * utility[next_key])
        delta += abs(utility[key] - u)
        utility[key] = u
        
      print(delta)
      if delta <= self.delta_convergence_threshold:
        break
    # print(utility)
    
  def build_policy(self, piece):
    debug = False
    did_change =  False
    for key in self.policy[piece]:
      if key == 'win-1' or key == 'win1':
        continue
      best_utility = -float('inf')
      best_neighbor = None
      debug = False
      if key == '000000000':
        debug = True
      if debug:
        print(self.policy[piece][key])
      for neighbor in self.state_map[key]:
        if debug:
          print("neighbor", neighbor, self.utilities[piece][neighbor])
        if self.utilities[piece][neighbor] > best_utility:
          best_neighbor = neighbor
          best_utility = self.utilities[piece][neighbor]
      if self.policy[piece][key] != best_neighbor:
        did_change = True
        self.policy[piece][key] = best_neighbor
    return did_change
      
class HumanStrategy(TicTacToeStrategy):
  def __init__(self, *args, **kwargs):
    super().__init__(t='human', *args, **kwargs)

  def build(self):
    pass

  def process_state(self, state, reward=0):
    invalid = True
    while invalid:
      
      in_coord = input()
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
  
    def _test_grid(self):
      # g = TicTacToeState(3)
      # g.do_move(0,2,1)
      # g2 = TicTacToeState(3)
      # g2.do_move(2,2,1)
      # self.assertEqual(g.state_key(), g2.state_key())
      # g2.do_move(0,2,1)
      # self.assertNotEqual(g.state_key(), g2.state_key())
      pass
    def test_game(self):
      t = TicTacToe(3)
      t.build_state_map()
      player1_strategy = TestStrategy(t='human')
      
      player2_strategy = TestStrategy(t='agent')
      
      player1_strategy.moves.append((0,2))
      player2_strategy.moves.append('000-100100')
      player1_strategy.moves.append((1,1))
      player2_strategy.moves.append('0-1101-1000')
      player1_strategy.moves.append((2,0))
      t.start_game(player1 = player1_strategy, player2 = player2_strategy)
      self.assertEqual(t.state.state_key(), '0-1101-1100')
      
  # unittest.main()
  t = TicTacToe(3)
  t.build_state_map(force=False)
  player1 = PolicyIterationStrategy(state_map = t.state_map)
  player1.build()
  
  player2 = HumanStrategy()
  t.start_game(player1 = player1, player2 = player2)
  #all winning states
  # print("Winners")
  # win_counter = {-1:0, 1:0}
  # for key in t.state_map:
  #   if 'win1' in t.state_map[key]:
  #     print(key)
  #     TicTacToeState.create(key, 3, win_counter=win_counter)
  # print(win_counter)