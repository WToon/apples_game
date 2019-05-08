import random
def get_random_decision(player, players, apples):
    nm = random.choices(['move','left','right'])
    return nm[0]