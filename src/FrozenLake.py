'''
author: patrick miller
gihub: github.com/patrickdmiller
gatech: pmiller75
'''

from GridMDP import GridMDP, PolicyIterationStrategy, ValueIterationStrategy, QLearnerStrategy
from gym import make as gymmake
# from gym.envs import register
from gym.envs.toy_text.frozen_lake import generate_random_map
import pickle as pk
import os
import numpy as np
# register("FrozenLake-v1", max_episode_steps=250)

class FrozenLake:
  def __init__(self, map_pickle_dir = './', strategy=PolicyIterationStrategy):
    self.map_pickle_dir = map_pickle_dir
    self.maps = {}
    if os.path.isfile(os.path.join(map_pickle_dir, 'maps_pickle.p')):
      #load
      print("loading maps")
      with open(os.path.join(map_pickle_dir, 'maps_pickle.p'), 'rb') as pickle_file:
        self.maps = pk.load(pickle_file)
    else:
      self.generate_and_save_maps()
    
    self.mdp = None
    #by default set it to the first map
    self.set_active_map(list(self.maps.keys())[0], 0)
    
    self.action_convert = {0:3, 1:2, 2:1, 3:0} #frozen lake 0 is left, and gridmdp 3 is left
    self.strategy = strategy
    
  def init_mdp(self, default_reward = 0):
    self.mdp = GridMDP(w = self.map_size, h=self.map_size, motion_forward_prob=1/3, motion_side_prob=1/3, motion_back_prob=0, default_reward=default_reward)
    # self.mdp = GridMDP(w = self.map_size, h=self.map_size, motion_forward_prob=1, motion_side_prob=0, motion_back_prob=0, default_reward=default_reward)

    for r in range(self.map_size):
      for c in range(self.map_size):
        if self.map[r][c] == 'G':
          self.mdp.grid.add_reward(p=(r,c), amt=10000)
          self.mdp.grid.add_terminal(p=(r,c))
        elif self.map[r][c] == 'H':
          #is a hole invalid or negative reward? You don't bounce off. you actually end the game so i am going with negative reward here. 
          # self.mdp.grid.add_invalid(p=(r,c))
          self.mdp.grid.add_terminal(p=(r,c))
          self.mdp.grid.add_reward(p=(r,c), amt=-1)
          
    self.mdp.compute_state_transition_matrix()
    
  def generate_and_save_maps(self, num_per_size=10, sizes=[4,8,20], **kwargs):
    for s in sizes:
      self.maps[s] = []
      for n in range(num_per_size):
        self.maps[s].append(generate_random_map(size=s, **kwargs))
    with  open(os.path.join(self.map_pickle_dir, 'maps_pickle.p'), 'wb') as pickle_file:
      pk.dump(self.maps,pickle_file)
  
  def set_active_map(self, key, index):
    self.map_size = key
    self.map = self.maps[key][index]
    # self.init_mdp(default_reward=-0.04)
    self.init_mdp(default_reward=-.04)
  
  def add_observation(self, observation, info, reward=None):
    # observation, info = dat
    action = self.mdp.process_state(s=observation, reward=reward)
    # print("at position", observation, "action = ", action)    
    return self.action_convert[action]

if __name__ == '__main__':
  fl = FrozenLake()
  fl.generate_and_save_maps(sizes=[8], p=0.9)
  fl.set_active_map(key=8, index=0)
  env = gymmake("FrozenLake-v1", desc=fl.map, is_slippery=True, max_episode_steps=2000, render_mode="")

  # fl.mdp.build_policy(strategy=PolicyIterationStrategy)
  fl.env = env
  fl.mdp.build_policy(strategy=QLearnerStrategy, environment = fl)
  # print("UTILITY: ", fl.mdp.strategy.utilities.values)


  i=0
  
  history = []
  runs = 0
  first_move= True
  # env.render_mode("human")
  # env = gymmake("FrozenLake-v1", desc=fl.map, is_slippery=True, max_episode_steps=2000, render_mode="human")

  print(env.spec)
  while runs < 100:
    if first_move:
      #this is the first move we make.
      observation, info = env.reset()
      action = fl.add_observation(observation, info)
      terminated, truncated = False, False
      first_move = False
    else:
      observation, reward, terminated, truncated, info = env.step(action)
      action = fl.add_observation(observation, info, reward)
    if terminated or truncated:
        # print(observation, reward, terminated, truncated, info)
        # print(fl.mdp.strategy.pp(which=fl.mdp.strategy.Q))
        # observation, info = env.reset()
        fl.mdp.clear_policy_memory()
        # action = fl.add_observation(observation, info)
        history.append([reward, i])
        runs+=1
        i=0
        first_move = True
    else:
      i+=1   
         
  
  #win/loss
  history = np.array(history)
  print("win", np.sum(history[:, 0]), "of ", len(history))
  print("avg moves", np.average(history[:,1]))
  env.close()
  # print(fl.mdp.strategy.pp(fl.map))
  for r in range(len(fl.map)):
    for c in range(len(fl.map[r])):
      print(fl.map[r][c], end="\t")
    print("")