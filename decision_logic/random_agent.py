import random


class RandomAgent:

  def next_action(self,players, apples):
    nm = random.choices(['move','left','right','fire'])
    return nm[0]
