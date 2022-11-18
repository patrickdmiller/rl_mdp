from FrozenLake import FrozenLake
from GridMDP import ValueIterationStrategy, PolicyIterationStrategy, QLearnerStrategy

fl = FrozenLake()
fl.load_saved_maps()
# map_sizes = [4,8]
# fl.generate_and_save_maps(sizes=map_sizes, p=0.9, num_per_size=2)
# fl.run(strategy=QLearnerStrategy)
# # fl.set_active_map(key=8, index=0)
# # env = gymmake("FrozenLake-v1", desc=fl.map, is_slippery=True, max_episode_steps=2000, render_mode="")
results = {}
results['VI'] = fl.run(strategy=ValueIterationStrategy)
results['PI'] = fl.run(strategy=PolicyIterationStrategy)
results['Q'] = fl.run(strategy=QLearnerStrategy)
for m in fl.maps:
  print('VI', results['VI'][m]['summary'])
  print('PI', results['PI'][m]['summary'])
  print('Q', results['Q'][m]['summary'])