import random
from decision_logic import dqn
from decision_logic import greedy_agent

NB_AGENTS = 3
NB_APPLES = 5
NB_TURNS = 100


class AGTrainingEnvironmentMPAgents:

    def __init__(self, nb_apples=NB_APPLES, nb_agents=NB_AGENTS):
        """
        An 'apples game' training environment, initialized with random agents
        :param nb_apples: The initial number of apples
        :param nb_agents: The number of agents
        """
        self.apples = self._init_apples(nb_apples)                      # list of apples in the game
        self.worms = self._init_worms(nb_agents)                        # list of worms in the game
        self.agents = [greedy_agent.GreedyAgent()]
        while len(self.agents) < nb_agents:
            self.agents.append(greedy_agent.GreedyAgent())

    def _play_turn(self):
        # Have each worm play a turn
        for index in range(len(self.worms)):
            worm = self.worms[index]
            location = worm['location']

            # Generate list of visible worms
            visible_players = []
            for _worm in self.worms:
                if self._is_visible_pos(location,_worm['location']):
                    visible_players.append(_worm)
                else:
                    visible_players.append({'location': ["?","?"], 'orientation': ["?","?"], 'score': _worm['score']})
            # Generate list of visible apples
            visible_apples = []
            for apple in self.apples:
                if self._is_visible_pos(location,apple):
                    visible_apples.append(apple)
            action = self.agents[index].next_action(index+1, visible_players, visible_apples)
            self.worms[index] = self._resolve_action(index,action)
        self._tick_apples()

    def _tick_apples(self):
        """
        An apple spawn cycle
        """
        for x in range(1,37):
            for y in range(1,17):
                if not [x,y] in self.apples:
                    nearby_apples = self._get_nearby_apples(x,y)
                    if nearby_apples == 0:
                        P = 0
                    elif nearby_apples == 1:
                        P = 0.005
                    elif nearby_apples == 2 :
                        P = 0.02
                    else:
                        P = 0.05
                    if random.uniform(0,1) < P:
                        self.apples.append([x,y])

    def _resolve_action(self,worm_index,action):
        worm = self.worms[worm_index]
        x, y = worm['location']
        orientation = worm['orientation']
        score = worm['score']

        if action == "move":
            next_orientation = orientation
            if orientation == "up":
                new_x = x
                new_y = (y-1) % 16
            elif orientation == "left":
                new_x = (x-1) % 36
                new_y = y
            elif orientation == "right":
                new_x = (x+1) % 36
                new_y = y
            elif orientation == "down":
                new_x = x
                new_y = (y+1) % 16

        elif action == "left":
            if orientation == "up":
                next_orientation = "left"
                new_x = (x-1) % 36
                new_y = y
            elif orientation == "left":
                next_orientation = "down"
                new_x = x
                new_y = (y+1) % 16
            elif orientation == "right":
                next_orientation = "up"
                new_x = x
                new_y = (y-1) % 16
            elif orientation == "down":
                next_orientation = "right"
                new_x = (x+1) % 36
                new_y = y
        elif action == "right":
            if orientation == "up":
                next_orientation ="right"
                new_x = (x+1) % 36
                new_y = y
            elif orientation == "left":
                next_orientation = "up"
                new_x = x
                new_y= (y-1) % 16
            elif orientation == "right":
                next_orientation= "down"
                new_x = x
                new_y= (y+1) % 16
            elif orientation == "down":
                next_orientation = "left"
                new_x = (x-1) % 36
                new_y = y

        if new_x == 0:
            new_x = 36
        if new_y == 0:
            new_y = 16
        if [new_x,new_y] in self.apples:
            score += 1
            self.apples.remove([new_x,new_y])
        return {'location':[new_x,new_y], 'orientation':next_orientation, 'score':score}

    def _play_game(self,gameID, verbose=False):
        turn = 0
        while len(self.apples) != 0 and turn != NB_TURNS:
            turn += 1
            self._play_turn()
        self.agents[0].save()
        if verbose:
            scores = [worm['score'] for worm in self.worms]
            print("Game {}\n Turns: {}\nScores:\n  TrainingAgent: {}\n  Others       : {}"
                  .format(gameID+1, turn, scores[0], scores[1:]))

    def play(self, nb_games=100):
        for gameID in range(nb_games):
            self._play_game(gameID)
            self._reset()
            progress_bar = ""
            for i in range(gameID+1): progress_bar += "#"
            for i in range(nb_games-(gameID+1)): progress_bar += "-"
            print(progress_bar)

    def _reset(self):
        self.__init__()

    @staticmethod
    def _init_apples(nb_apples):
        apples = []
        while len(apples) < nb_apples:
            apple = [random.randint(1, 36), random.randint(1, 16)]
            if apple not in apples:
                apples.append(apple)
        return apples

    @staticmethod
    def _init_worms(nb_agents):
        agent_locations = []
        # Generate unique locations for each training_agent
        while len(agent_locations) < nb_agents:
            agent_location = [[random.randint(1, 36), random.randint(1, 16)]]
            if agent_location not in agent_locations:
                agent_locations.append(agent_location)
        # Generate orientations for each training_agent
        agent_orientations = [[random.choice(['up', 'right', 'left', 'down'])] for i in range(0, nb_agents)]

        worms = zip(agent_locations, agent_orientations)
        agents = [{'location': w[0][0], 'orientation': w[1][0], 'score': 0} for w in worms]
        return agents

    @staticmethod
    def _distance(a, b):
        dx = abs(a[0] - b[0])
        if dx > 36/2:
            dx = 36 - dx;
        dy = abs(a[1] - b[1])
        if dy > 16/2:
            dy = 16 - dy;
        return (dx * dx) + (dy * dy);

    @staticmethod
    def _is_visible_pos(pos1,pos2):
        dx = abs(pos1[0] - pos2[0])
        if dx > 18:
            dx = 36-dx
        dy = abs(pos1[1]-pos2[1])
        if dy > 8:
            dy = 16-dy
        if dx <= 7:
            if dy <= 7:
                return True
        return False

    def _get_nearby_apples(self,x,y):
        """
        :param x: Current x-coord
        :param y: Current y-coord
        :return: The amount of nearby apples
        """
        nb = 0
        for nbx in range(-2,3):
            for nby in range(-2,3):
                if nbx !=0 or nby != 0:
                    newx = (x+nbx)%36
                    newy = (y+nby)%16
                    if newx == 0:
                      newx = 36
                    if newy == 0:
                      newy = 16
                    if self._distance([x, y], [newx, newy]) <= 2 and [newx,newy] in self.apples:
                        nb+=1
        return nb


Env = AGTrainingEnvironmentMPAgents()
Env.play(100)