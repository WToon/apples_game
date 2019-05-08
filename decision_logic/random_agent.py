import random
import numpy as np
class RandomAgent:
  def __init__(self):
    self.actions=np.zeros(4)

  def next_action(self,players, apples):
    nm = random.choices(['move','left','right','fire'])
    self.actions[['move','left','right','fire'].index(nm[0])]+=1
    return nm[0]
