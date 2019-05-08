import random, itertools
from decision_logic import dqn
from decision_logic import random_agent as ra

NB_AGENTS = 5
NB_APPLES = 13
NB_TURNS=100


class AGTrainingEnvironment:

    def __init__(self, nb_apples=NB_APPLES, nb_agents=NB_AGENTS):
        """
        An 'apples game' training environment, initialized with random agents
        :param nb_apples: The initial number of apples
        :param nb_agents: The number of agents
        """
        self.apples = self._init_apples(nb_apples)                      # list of apples in the game
        self.worms = self._init_worms(nb_agents)                        # list of worms in the game
        self.agent = dqn.DQNAgent()

    @staticmethod
    def _init_apples(nb_apples):
        apples = [[random.randint(1,36), random.randint(1,16)] for i in range(0,nb_apples)]
        apples.sort()
        apples = [k for k,_ in itertools.groupby(apples)]
        return apples

    @staticmethod
    def _init_worms(nb_agents):
        agent_locations = []
        # Generate unique locations for each agent
        while len(agent_locations) < nb_agents:
            agent_locations= [[[random.randint(1, 36), random.randint(1, 16)]] for i in range(0, nb_agents)]
            agent_locations.sort()
            agent_locations= [k for k, _ in itertools.groupby(agent_locations)]
        # Generate orientations for each agent
        agent_orientations = [[random.choice(['up', 'right', 'left', 'down'])] for i in range(0, nb_agents)]

        worms = zip(agent_locations, agent_orientations)
        agents = [{'location': w[0][0], 'orientation': w[1][0], 'score': 0} for w in worms]
        return agents

    def _play_turn(self,verbose=True):
        for i in range(0,len(self.worms)):
            agent_worm = self.worms[i]
            agent_pos = agent_worm['location']
            players = []
            for _worm in self.worms:
                if self._is_visible_pos(agent_pos,_worm['location']):
                    players.append(_worm)
                else:
                    players.append({'location': ["?","?"], 'orientation': ["?","?"], 'score': _worm['score']})
            apples = []
            for apple in self.apples:
                if self._is_visible_pos(agent_pos,apple):
                    apples.append(apple)
            action = self.agent.next_action(i+1,players,apples)
            print("Worm,Action : {}, {}".format(agent_worm, action))
            self.worms[i] = self._resolve_action(i,action,self.worms)
        self._add_apples()
        if verbose:
            print("Apples: {}".format(self.apples))
            print("Players: {}".format(self.worms))


    def _is_visible_pos(self,pos1,pos2):
      dx = abs(pos1[0] - pos2[0])
      if dx>18:
        dx = 36-dx
      dy = abs(pos1[1]-pos2[1])
      if dy>8:
        dy = 16-dy
      if (dx<=7):
        if (dy<=7):
          return True
      return False

    def _add_apples(self):
        for x in range(1,37):
            for y in range(1,17):
                if not [x,y] in self.apples:
                    nbNeighbours = self._get_nearby_apples(x,y)
                    P = None
                    if nbNeighbours == 0:
                        P = 0
                    elif nbNeighbours == 1:
                        P = 0.005
                    elif nbNeighbours == 2 :
                        P = 0.02
                    else:
                        P = 0.05
                    if (random.uniform(0,1) < P):
                        self.apples.append([x,y])

    def _get_nearby_apples(self,x,y):
        nbNeighbors = 0;
        for nbx in range(-2,3):
            for nby in range(-2,3):
                if nbx !=0 or nby != 0:
                    newx = (x+nbx)%36
                    newy = (y+nby)%16
                    if newx ==0:
                      newx = 36
                    if newy ==0:
                      newy = 16
                    if self._distance([x, y], [newx, newy]) <= 2 and [newx,newy] in self.apples:
                        nbNeighbors+=1
        return nbNeighbors

    def _distance(self,a, b):
        dx = abs(a[0] - b[0])
        if (dx > 36 / 2):
            dx = 36 - dx;
        dy = abs(a[1] - b[1])
        if (dy > 16 / 2):
            dy = 16 - dy;
        return (dx * dx) + (dy * dy);

    def _resolve_action(self,index,action,players):
      worm = self.worms[index]
      location=worm['location']
      x=location[0]
      y=location[1]
      orientation = worm['orientation']
      score = worm['score']
      if action == "move":
        neworientation = orientation
        if orientation=="up":
          newx=x
          newy = (y-1)%16
        elif orientation =="left":
          newx = (x-1)%36
          newy=y
        elif orientation == "right":
          newx = (x+1)%36
          newy=y
        else:
          newx = x
          newy = (y+1)%16

      elif action == "left":
        if orientation=="up":
          neworientation ="left"
          newx=(x-1)%36
          newy = y
        elif orientation =="left":
          neworientation ="down"
          newx = x
          newy=(y+1)%16
        elif orientation == "right":
          neworientation ="up"
          newx = x
          newy=(y-1)%16
        else:
          neworientation ="right"
          newx = (x+1)%36
          newy = y
      elif action=="right":
        if orientation=="up":
          neworientation ="right"
          newx=(x+1)%36
          newy = y
        elif orientation =="left":
          neworientation ="up"
          newx = x
          newy=(y-1)%16
        elif orientation == "right":
          neworientation ="down"
          newx = x
          newy=(y+1)%16
        else:
          neworientation ="left"
          newx = (x-1)%36
          newy = y
      else:
        neworientation =orientation
        newx=x
        newy=y
        self._zap(x,y,orientation,players)
        score -=1
      if newx ==0:
        newx = 36
      if newy ==0:
        newy = 16
      if [newx,newy] in self.apples:
        score+=1
        self.apples.remove([newx,newy])
      return {'location':[newx,newy], 'orientation':neworientation, 'score':score}

    def _zap(self,x,y,orientation,players):
      zapped=False
      for i in range(1,8):
        if orientation=='up':
          newx = x
          newy = (y-i)%16
        elif orientation=='left':
          newx = (x-i)%36
          newy=y
        elif orientation=='right':
          newx = (x+i)%36
          newy = y
        else:
          newx = x
          newy = (y + i) % 16
        if newx == 0:
          newx=36
        if newy==0:
          newy = 16
        if not zapped:
          for player in players:
            if player['location'] == [newx,newy] and not (newx == x and newy == y):
              player['score'] -= 50
              zapped=True


    def play(self, turns=NB_TURNS):
      turn = 0
      print("BEGINNING THE GAME")
      while len(self.apples)!=0 and turn != turns:
        turn +=1
        print("TURN ", turn)
        self._play_turn(verbose=True)
      print("GAME HAS ENDED")


Env = AGTrainingEnvironment()
Env.play(10)
