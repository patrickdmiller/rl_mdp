from FrozenLake import FrozenLake
from GridMDP import ValueIterationStrategy, PolicyIterationStrategy, QLearnerStrategy
import pickle as pk
import os
fl = FrozenLake()
fl.load_saved_maps()
# map_sizes = [4,20]
# fl.generate_and_save_maps(sizes=map_sizes, p=0.9, num_per_size=10)
# fl.run(strategy=QLearnerStrategy)
# # fl.set_active_map(key=8, index=0)
# # env = gymmake("FrozenLake-v1", desc=fl.map, is_slippery=True, max_episode_steps=2000, render_mode="")

# results['VI']['gamma_0.5'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.5)
# results['VI']['gamma_0.7'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.7)
# results['VI']['gamma_0.9'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.9)
def do_pickle(results, param):     
  with  open(os.path.join('./', f'results_{param}_pickle.p'), 'wb') as pickle_file:
        pk.dump(results,pickle_file)

def do_q():
  results = {'Q':{
  'gamma_test':{}, 'alpha_test':{}, 'epsilon_test':{}, 'punish_test':{}}, 'VI':{}, 'PI':{}}
  results['Q']['punish_test']['gamma_0.9|alpha_0.5|epsilon_0.05|punish_false'] = fl.run(strategy=QLearnerStrategy, gamma=0.9, alpha=0.5, epsilon = 0.05, runs_per_map=2, max_iterations_after_finding_goal_factor=100, punish = False)
  results['Q']['punish_test']['gamma_0.9|alpha_0.5|epsilon_0.05|punish_true'] = fl.run(strategy=QLearnerStrategy, gamma=0.9, alpha=0.5, epsilon = 0.05, runs_per_map=100, max_iterations_after_finding_goal_factor=100, punish = True)


  #best was .9 gamma
  results['Q']['gamma_test']['gamma_0.5|alpha_0.5|epsilon_0.05|punish_true'] = fl.run(strategy=QLearnerStrategy, gamma=0.5, alpha=0.5, epsilon = 0.05, runs_per_map=100, max_iterations_after_finding_goal_factor=100, punish = True)
  results['Q']['gamma_test']['gamma_0.7|alpha_0.5|epsilon_0.05|punish_true'] = fl.run(strategy=QLearnerStrategy, gamma=0.7, alpha=0.5, epsilon = 0.05, runs_per_map=100, max_iterations_after_finding_goal_factor=100, punish = True)
  results['Q']['gamma_test']['gamma_0.9|alpha_0.5|epsilon_0.05|punish_true'] = fl.run(strategy=QLearnerStrategy, gamma=0.9, alpha=0.5, epsilon = 0.05, runs_per_map=100, max_iterations_after_finding_goal_factor=100, punish = True)

  #best is .5 alpha
  results['Q']['alpha_test']['gamma_0.9|alpha_0.25|epsilon_0.05|punish_true'] = fl.run(strategy=QLearnerStrategy, gamma=0.9, alpha=0.25, epsilon = 0.05, runs_per_map=100, max_iterations_after_finding_goal_factor=100, punish = True)
  results['Q']['alpha_test']['gamma_0.9|alpha_0.5|epsilon_0.05|punish_true'] = fl.run(strategy=QLearnerStrategy, gamma=0.9, alpha=0.5, epsilon = 0.05, runs_per_map=100, max_iterations_after_finding_goal_factor=100, punish = True)
  results['Q']['alpha_test']['gamma_0.9|alpha_0.9|epsilon_0.05|punish_true'] = fl.run(strategy=QLearnerStrategy, gamma=0.9, alpha=0.9, epsilon = 0.05, runs_per_map=100, max_iterations_after_finding_goal_factor=100, punish = True)

  results['Q']['epsilon_test']['gamma_0.9|alpha_0.5|epsilon_0.05|punish_true'] = fl.run(strategy=QLearnerStrategy, gamma=0.9, alpha=0.5, epsilon = 0.05, runs_per_map=100, max_iterations_after_finding_goal_factor=100, punish = True)
  results['Q']['epsilon_test']['gamma_0.9|alpha_0.5|epsilon_0.15|punish_true'] = fl.run(strategy=QLearnerStrategy, gamma=0.9, alpha=0.5, epsilon = 0.15, runs_per_map=100, max_iterations_after_finding_goal_factor=100, punish = True)
  results['Q']['epsilon_test']['gamma_0.9|alpha_0.5|epsilon_0.25|punish_true'] = fl.run(strategy=QLearnerStrategy, gamma=0.9, alpha=0.5, epsilon = 0.25, runs_per_map=100, max_iterations_after_finding_goal_factor=100, punish = True)

  return results

def do_vi():
  results = {'VI':{}}
  results['VI']['gamma_0.5|deltathreshold_1'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.5, delta_convergence_threshold=1)
  results['VI']['gamma_0.7|deltathreshold_1'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.7, delta_convergence_threshold=1)
  results['VI']['gamma_0.9|deltathreshold_1'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.9, delta_convergence_threshold=1)
  results['VI']['gamma_0.5|deltathreshold_0.1'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.5, delta_convergence_threshold=.1)
  results['VI']['gamma_0.7|deltathreshold_0.1'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.7, delta_convergence_threshold=.1)
  results['VI']['gamma_0.9|deltathreshold_0.1'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.9, delta_convergence_threshold=.1)
  results['VI']['gamma_0.5|deltathreshold_0.001'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.5, delta_convergence_threshold=.001)
  results['VI']['gamma_0.7|deltathreshold_0.001'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.7, delta_convergence_threshold=.001)
  results['VI']['gamma_0.9|deltathreshold_0.001'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.9, delta_convergence_threshold=.001)
  return results
def do_pi():
  results = { 'PI':{}}

  results['PI']['gamma_0.5|deltathreshold_1'] = fl.run(strategy=PolicyIterationStrategy, runs_per_map=100, gamma=0.5, delta_convergence_threshold=1)
  results['PI']['gamma_0.7|deltathreshold_1'] = fl.run(strategy=PolicyIterationStrategy, runs_per_map=100, gamma=0.7, delta_convergence_threshold=1)
  results['PI']['gamma_0.9|deltathreshold_1'] = fl.run(strategy=PolicyIterationStrategy, runs_per_map=100, gamma=0.9, delta_convergence_threshold=1)
  results['PI']['gamma_0.5|deltathreshold_0.1'] = fl.run(strategy=PolicyIterationStrategy, runs_per_map=100, gamma=0.5, delta_convergence_threshold=.1)
  results['PI']['gamma_0.7|deltathreshold_0.1'] = fl.run(strategy=PolicyIterationStrategy, runs_per_map=100, gamma=0.7, delta_convergence_threshold=.1)
  results['PI']['gamma_0.9|deltathreshold_0.1'] = fl.run(strategy=PolicyIterationStrategy, runs_per_map=100, gamma=0.9, delta_convergence_threshold=.1)
  results['PI']['gamma_0.5|deltathreshold_0.001'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.5, delta_convergence_threshold=.001)
  results['PI']['gamma_0.7|deltathreshold_0.001'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.7, delta_convergence_threshold=.001)
  results['PI']['gamma_0.9|deltathreshold_0.001'] = fl.run(strategy=ValueIterationStrategy, runs_per_map=100, gamma=0.9, delta_convergence_threshold=.001)
  return results


do_pickle(do_vi(),'vi')
do_pickle(do_pi(), 'pi')

# for p in results:
#   for tags in results[p]:
#     for m in results[p][tags]:
#       print(p,tags,m,results[p][tags][m]['summary'])
