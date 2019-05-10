import random, itertools
nb_agents = 3

agent_locations = []
# Generate unique locations for each agent
while len(agent_locations) < nb_agents:
    agent_locations = [[[random.randint(1, 36), random.randint(1, 16)]] for i in range(0, nb_agents)]
    agent_locations.sort()
    agent_locations = [k for k, _ in itertools.groupby(agent_locations)]
# Generate orientations for each agent
agent_orientations = [[random.choice(['up', 'right', 'left', 'down'])] for i in range(0, nb_agents)]

worms = zip(agent_locations, agent_orientations)
agents = [{'location': w[0][0], 'orientation': w[1][0], 'score': 0} for w in worms]
print(agents)