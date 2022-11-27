from TicTacToe import TicTacToe, QLearningStrategy, PolicyIterationStrategy, ValueIterationStrategy, RandomStrategy, TicTacToeStateMap, TicTacToeState

import pickle as pk
import os




def do_pickle(results, param, name='ttt'):     
  with  open(os.path.join('./', f'ttt_{name}_results_{param}_pickle.p'), 'wb') as pickle_file:
        pk.dump(results,pickle_file)

def do_q(n=3, training_sizes=[1,10,100,1000,10000,100000, 500000], epsilons=[0.01,0.05,0.1,0.25], opponents = None, name=None):
  results = {}
  t = TicTacToe(n)
  # opponents = [RandomStrategy(), PolicyIterationStrategy(state_map = t.state_map), ValueIterationStrategy(state_map=t.state_map)]

  for i in training_sizes:
    for e in epsilons:
      results[f'e_{e}|c_{i}'], player = t.run(strategy=QLearningStrategy, opponents = opponents, max_learning_epochs=i, epsilon=e)
  if name is None:
    name = f'{n}'
  do_pickle(results, 'q', name)
  return results

def do_vi_pi(n=3, gammas = [0.5], delta_convergence_thresholds=[0.1], name=None):
  results = {'PI':{}, 'VI':{}}
  t = TicTacToe(n)
  for g in gammas:
    for d in delta_convergence_thresholds:
      print("g",g,"d",d)
      results['PI'][f'g_{g}|d_{d}'], player = t.run(strategy=PolicyIterationStrategy, opponents=None, delta_convergence_threshold = d, gamma = g, num_games=10000)
      results['VI'][f'g_{g}|d_{d}'], player = t.run(strategy=ValueIterationStrategy, opponents=None, delta_convergence_threshold = d, gamma = g, num_games=10000)
      
  if name is None:
    name = f'{n}'
  # print(results)
  do_pickle(results, 'vipi', name)

# do_q(n=3, training_sizes=[1,10,100,1000, 10000], name='epsilon_test_3')
# do_q(n=4, training_sizes=[1,10,100,1000, 10000, 100000, 500000], name='epsilon_test_4')
do_vi_pi(n=3, gammas = [0.5], delta_convergence_thresholds=[0.01,0.1,1,10], name="3_just_d")
# do_vi_pi(n=4, gammas = [0.5], delta_convergence_thresholds=[0.01,0.1,1,10], name="4_just_d")