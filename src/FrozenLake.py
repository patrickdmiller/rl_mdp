'''
author: patrick miller
gihub: github.com/patrickdmiller
gatech: pmiller75
'''

from GridMDP import GridMDP
from gym import make as gymmake
from gym.envs.toy_text.frozen_lake import generate_random_map
import pickle as pk
import os


class FrozenLake:
  def __init__(self, map_pickle_dir = './'):
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
  def init_mdp(self):
    self.mdp = GridMDP(w = self.map_size, h=self.map_size, motion_forward_prob=1/3, motion_side_prob=1/3, motion_back_prob=0)
    for r in range(self.map_size):
      for c in range(self.map_size):
        if self.map[r][c] == 'G':
          self.mdp.grid.add_reward(p=(r,c), amt=1)
        elif self.map[r][c] == 'H':
          self.mdp.grid.add_invalid(p=(r,c))
    self.mdp.compute_state_transition_matrix()
    
  def generate_and_save_maps(self, num_per_size=2, sizes=[4,20]):
    for s in sizes:
      self.maps[s] = []
      for n in range(num_per_size):
        self.maps[s].append(generate_random_map(size=s))
    with  open(os.path.join(self.map_pickle_dir, 'maps_pickle.p'), 'wb') as pickle_file:
      pk.dump(self.maps,pickle_file)
  
  def set_active_map(self, key, index):
    self.map_size = key
    self.map = self.maps[key][index]
    self.init_mdp()
  
  def add_observation(self, observation, info):
    # observation, info = data
    action = self.mdp.get_action(s=observation)
    return self.action_convert[action]
if __name__ == '__main__':
  fl = FrozenLake()
  fl.set_active_map(key=4, index=0)
  fl.mdp.build_policy()

  env = gymmake("FrozenLake-v1", desc=fl.map, is_slippery=False, render_mode="human")
  observation, info = env.reset(seed=42)
  action = fl.add_observation(observation, info)
  
  for _ in range(1000):
    
    observation, reward, terminated, truncated, info = env.step(action)
    action = fl.add_observation(observation, info)

    if terminated or truncated:
        observation, info = env.reset()

  env.close()