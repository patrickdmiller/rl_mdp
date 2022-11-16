
class Utility:
  def __init__(self, w, h):
    self.values = {}
    self.h, self.w = h,w
    for r in range(self.h):
      for c in range(self.w):
        self.values[(r, c)] = 0

  def get(self, p):
    if p in self.values:
      return self.values[p]
    return 0
  def __getitem__(self, p):
    if p in self.values:
      return self.values[p]
    return 0
  def set(self, p, value):
    if p not in self.values:
      raise Exception("invalid point in utility", p)
    self.values[p] = value
  
#0 is up, 1 is right
class Directions:
  def __init__(self, grid):
    self.dir_transitions = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    self.probs = {}
    self.length = 4
    self.grid = grid

  def get_coordinate_from_direction(self, p, direction):
    assert direction < len(self.dir_transitions) and direction >= 0
    return (p[0] + self.dir_transitions[direction][0],
                 p[1] + self.dir_transitions[direction][1])

  def set_probabilities(self, f=1, s=0, b=0):
    self.probs['f'] = f
    self.probs['s'] = s
    self.probs['b'] = b
  
  #returns neighbor and direction
  def get_all_neighbors_and_directions(self, p):
    neighbors = []
    for d in range(len(self)):
      neighbor = self.get_coordinate_from_direction(p, d)
      neighbors.append((neighbor, d))
    return neighbors
  
  def get_valid_neighbors_and_directions(self, p):
    neighbors = []
    for d in range(len(self)):
      neighbor = self.get_coordinate_from_direction(p, d)
      if self.grid.is_valid(neighbor):
        neighbors.append((neighbor, d))
    return neighbors
  
  #return coordinates and probabilities that we would try and go to
  def get_target_coordinate_and_prob_from_direction(self, p, direction):
    ret = []
    if self.probs['f'] > 0:
      ret.append((self.get_coordinate_from_direction(p, direction), self.probs['f']))
    if self.probs['s'] > 0:
      #ccw
      ccw_direction = (direction - 1) % len(self)
      ret.append((self.get_coordinate_from_direction(p, ccw_direction), self.probs['s']))
      #cw
      cw_direction = (direction + 1) % len(self)
      ret.append((self.get_coordinate_from_direction(p, cw_direction), self.probs['s']))
    if self.probs['b'] > 0:
      ret.append((self.get_coordinate_from_direction(p, (direction + 2) % len(self)), self.probs['b']))
    return ret
  
  #return actual coordinates and probabilities that we would go to
  def get_resulting_coordinate_and_prob_from_direction(self, p, direction):
    results = []
    targets = self.get_target_coordinate_and_prob_from_direction( p, direction)
    
    # [((2, 4), 0.8), ((1, 3), 0.1), ((3, 3), 0.1)]
    for coord, prob in targets:
      if not self.grid.is_valid(coord):
        results.append((p, prob))
      else:
        results.append((coord, prob))
    return results
  def __len__(self):
    return self.length
