'''
generate maps once so we have consistency through experiments
'''

from FrozenLake import FrozenLake
fl = FrozenLake()
map_sizes = [4,20]
fl.generate_and_save_maps(sizes=map_sizes, p=0.9, num_per_size=10)
