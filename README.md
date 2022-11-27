repo : https://github.com/patrickdmiller/rl_mdp

## N-Sized TIC-TAC-TOE (N-TTT)
+ src/TicTacToe.py contains all classes
  + main TicTacToe class takes a strategy when run or playing a game. There are 5 strategies included
    + PolicyIteration, ValueIteration, QLearning, Random, Human
    + Human lets you play against one of the other strategies. See unit tests for example of injecting a player into a game with a specific strategy.
+ src/experiments_ttt.py contains various experiments run
+ if you're running this for the first time and n=4 it is time consuming as the entire state-graph must be built.

## Frozen Lake
+ src/FrozenLake.py - specifics for frozen lake using GYM and GridMDP.py
+ src/GridMDP.py - Generic Grid MDP classes
+ src/utilities.py - directions and getting stochastic moves from the grid
+ src/experiments_fl.py. contains various experiments run

src/results.ipynb builds charts and tables


TODO: a lot of cleanup, but need to hand in.