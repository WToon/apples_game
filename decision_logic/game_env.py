import random, itertools
from decision_logic import dqn

NB_AGENTS = 5
NB_APPLES = 13
NB_TURNS=100
class AG_training_environment():

    def __init__(self, nb_apples=NB_APPLES, nb_agents=NB_AGENTS):
        self.apples = self._init_apples(nb_apples)                  # list of apples in the game
        self.worms = self._init_worms(nb_agents)                    # list of worms in the game
        self.agents = [dqn.DQNAgent() for i in range(0,nb_agents)]  # list of agents (map to worms with index)

    @staticmethod
    def _init_apples(nb_apples):
        apples = [[random.randint(1,36), random.randint(1,16)] for i in range(0,nb_apples)]
        apples.sort()
        apples = [k for k,_ in itertools.groupby(apples)]
        return apples

    @staticmethod
    def _init_worms(nb_agents):
        agent_locations= []
        # Generate unique locations for each agent
        while len(agent_locations) < nb_agents:
            agent_locations= [[[random.randint(1, 36), random.randint(1, 16)]] for i in range(0, nb_agents)]
            agent_locations.sort()
            agent_locations= [k for k, _ in itertools.groupby(agent_locations)]
        # Generate orientations for each agent
        agent_orientations = [[random.choice(['up', 'right', 'left', 'down'])] for i in range(0, nb_agents)]

        worms = zip(agent_locations, agent_orientations)
        _w = [{'location': w[0][0], 'orientation': w[1][0], 'score': 0} for w in worms]
        agents = {};i = 0
        for w in _w:
            agents[str(i)] = w
            i += 1
        return agents

    def turn(self):
      for index in range(0,NB_AGENTS):
        agent = self.worms.get(str(index))
        ownPosition = agent.get('location')
        players = []
        for worm in self.worms.values():
          position = worm.get('location')
          if self.visiblePosition(ownPosition,position):
            players.append(worm)
          else:
            players.append([{'location': ["?","?"], 'orientation': ["?","?"], 'score': worm.get('score')}])
        apples = []
        for apple in self.apples:
          if self.visiblePosition(ownPosition,apple):
            apples.append(apple)
        action = self.agents[index].next_action(players,apples)
        agent['location'], agent['orientation'] , agent['score'] = self.resolveAction(index,action,players)

    def visiblePosition(self,pos1,pos2):
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

    #add apples to nearby positions (same as in the game)
    def addApples(self):
        for x in range(1,37):
            for y in range(1,17):
                if not [x,y] in self.apples:
                    nbNeighbours = self.getApplesInNeighbourhood(x,y)
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


    def getApplesInNeighbourhood(self,x,y):
        nbNeighbors = 0;
        for nbx in range(-2,3):
            for nby in range(-2,3):
                if nbx !=0 or nby != 0:
                    newx = (x+nbx)%36+1
                    newx = (x+nbx)%36+1
                    newy = (x+nby)%16+1
                    if self.distance([x, y], [newx, newy]) <= 2 and [newx,newy] in self.apples:
                        nbNeighbors+=1
        return nbNeighbors

    def distance(self,a, b):
        dx = abs(a[0] - b[0])
        if (dx > 36 / 2):
            dx = 36 - dx;
        dy = abs(a[1] - b[1])
        if (dy > 16 / 2):
            dy = 16 - dy;
        return (dx * dx) + (dy * dy);

    def resolveAction(self,index,action,players):
      worm = self.worms.values()[index]
      location=worm.get('location')
      x=location[0]
      y=location[1]
      orientation = worm.get('orientation')
      score = worm.get('score')
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
        self.zap(x,y,orientation,players)
        score -=1
      if newx ==0:
        newx = 36
      if newy ==0:
        newy = 16
      if [newx,newy] in self.apples:
        score+=1
        self.apples.remove([newx,newy])
      return([newx,newy],neworientation ,score)

    def zap(self,x,y,orientation,players):
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
            if player.get('location')==[newx,newy]:
              player['score'] -= 50
              zapped=True


    def playOneGame(self):
      turn = 0
      print("BEGIN THE GAME")
      while(len(self.apples)!=0 or turn == NB_TURNS):
        turn +=1
        print("TURN ", turn)
        self.turn()
        self.addApples()
      print("GAME HAS ENDED")






Env = AG_training_environment()
Env.playOneGame()





    # ----------------------------------------------------------------------------
    # a worm does a certain action
    # def worm_step(self,worm,actionid):
    #     self.steps +=1
    #     print (self.steps)
    #     action = ACTIONS[actionid]
    #     x,y,orientation = worm.state
    #
    #     apples = worm.visibleApples
    #     newx=None
    #     newx=None
    #     newy=None
    #     neworientation =None
    #     orientation = ORIENTATION[orientation]
    #     reward = 0
    #     if action == "move":
    #         neworientation =ORIENTATION.index(orientation)
    #         if orientation=="up":
    #             newx=x
    #             newx=x
    #             newy = (y-1)%16
    #         elif orientation =="left":
    #             newx = (x-1)%36
    #             newx = (x-1)%36
    #             newy=y
    #         elif orientation == "right":
    #             newx = (x+1)%36
    #             newx = (x+1)%36
    #             newy=y
    #         else:
    #             newx = x
    #             newx = x
    #             newy = (y+1)%16
    #
    #     elif action == "left":
    #         if orientation=="up":
    #             neworientation =ORIENTATION.index("left")
    #             newx=(x-1)%36
    #             newx=(x-1)%36
    #             newy = y
    #         elif orientation =="left":
    #             neworientation =ORIENTATION.index("down")
    #             newx = x
    #             newx = x
    #             newy=(y+1)%16
    #         elif orientation == "right":
    #             neworientation =ORIENTATION.index("up")
    #             newx = x
    #             newx = x
    #             newy=(y-1)%16
    #         else:
    #             neworientation =ORIENTATION.index("right")
    #             newx = (x+1)%36
    #             newx = (x+1)%36
    #             newy = y
    #     else:
    #         if orientation=="up":
    #             neworientation =ORIENTATION.index("right")
    #             newx=(x+1)%36
    #             newx=(x+1)%36
    #             newy = y
    #         elif orientation =="left":
    #             neworientation =ORIENTATION.index("up")
    #             newx = x
    #             newx = x
    #             newy=(y-1)%16
    #         elif orientation == "right":
    #             neworientation =ORIENTATION.index("down")
    #             newx = x
    #             newx = x
    #             newy=(y+1)%16
    #         else:
    #             neworientation =ORIENTATION.index("left")
    #             newx = (x-1)%36
    #             newx = (x-1)%36
    #             newy = y
    #     if [newx, newy] in apples:
    #     if [newx, newy] in apples:
    #         reward +=1
    #         self.apples.remove([newx,newy])
    #         self.apples.remove([newx,newy])
    #         newApple = [random.randint(0,35),random.randint(0,36)]
    #         while newApple in self.apples:
    #             newApple = [random.randint(0, 35), random.randint(0, 36)]
    #         self.apples.append(newApple)
    #     newApples = self.checkApples(newx,newy,self.apples)
    #     newApples = self.checkApples(newx,newy,self.apples)
    #     return  ((newx,newy,neworientation ),newApples,reward)
    #     return  ((newx,newy,neworientation ),newApples,reward)
    #
    # #check which apples are visible for the agent
    # def checkApples(self,x,y,apples):
    #     result = []
    #     for i in apples:
    #         dx = abs(i[0]-x)
    #         if dx>18:
    #             dx = 36-dx
    #         dy = abs(i[1]-y)
    #         if dy>8:
    #             dy = 16-dy
    #         if (dx<=7):
    #             if (dy<=7):
    #                 result.append(i)
    #     return result
    #
    # #add apples to nearby positions (same as in the game)
    # def addApples(self):
    #     for x in range(0,36):
    #         for y in range(0,16):
    #             if not [x,y] in self.apples:
    #                 nbNeighbours = self.getApplesInNeighbourhood(x,y)
    #                 P = None
    #                 if nbNeighbours == 0:
    #                     P = 0
    #                 elif nbNeighbours == 1:
    #                     P = 0.005
    #                 elif nbNeighbours == 2 :
    #                     P = 0.02
    #                 else:
    #                     P = 0.05
    #                 if (random.uniform(0,1) < P):
    #                     self.apples.append([x,y])
    #
    #
    # def getApplesInNeighbourhood(self,x,y):
    #     nbNeighbors = 0;
    #     for nbx in range(-2,3):
    #         for nby in range(-2,3):
    #             if nbx !=0 or nby != 0:
    #                 newx = (x+nbx)%36
    #                 newx = (x+nbx)%36
    #                 newy = (x+nby)%16
    #                 if self.distance([x, y], [newx, newy]) <= 2 and [newx,newy] in self.apples:
    #                 if self.distance([x, y], [newx, newy]) <= 2 and [newx,newy] in self.apples:
    #                 if self.distance([x, y], [newx, newy]) <= 2 and [newx,newy] in self.apples:
    #                     nbNeighbors+=1
    #     return nbNeighbors
    #
    # def distance(self,a, b):
    #     dx = abs(a[0] - b[0])
    #     if (dx > 36 / 2):
    #         dx = 36 - dx;
    #     dy = abs(a[1] - b[1])
    #     if (dy > 16 / 2):
    #         dy = 16 - dy;
    #     return (dx * dx) + (dy * dy);
    #
    # #reset the worm (new position etc)
    # def wormreset(self,worm,apples):
    #     randomX = random.randint(0,35)
    #     randomY = random.randint(0,15)
    #     orientation=random.randint(0,3)
    #     worm.state = (randomX,randomY,orientation)
    #     worm.visibleApples = self.checkApples(randomX,randomY,apples)
    #
    # #reset the environment
    # def reset(self):
    #
    #     self.apples = []
    #     for i in range(0, NB_APPLES):
    #         newApple = [random.randint(0, 35), random.randint(0, 15)]
    #         while newApple in self.apples:
    #             newApple = [random.randint(0, 35), random.randint(0, 15)]
    #         self.apples.append(newApple)
    #     for worm in self.worms:
    #         self.wormreset(worm,self.apples)
    #
    # #take a step in the environment -> all worms do a move
    # def step(self,rewards):
    #     for worm in self.worms:
    #         x,y,orientation = worm.state
    #         worm.state = (x,y,orientation)
    #         worm.visibleApples = self.checkApples(x,y,self.apples)
    #         action = worm.act()
    #         newState,apples,reward = self.worm_step(worm,action)
    #         worm.train(worm.state,action,reward,newState)
    #         worm.state = newState
    #         worm.visibleApples = apples
    #         rewards[self.worms.index(worm)]+= reward
    #
    #     # self.addApples()
    #     return rewards
    #
    # #play one game
    # def runEpisode(self):
    #     self.reset()
    #     rewards = np.zeros(len(self.worms))
    #     for i in range(0,NB_TURNS):
    #         rewards = self.step(rewards)
    #     self.steps = 0
    #     print("DONE",rewards)
    #
    # #play a number of games after training
    # def runEpisodeAfterTraining(self):
    #     for x in range(0,EPISODES_AFTER_TRAINING):
    #         self.reset()
    #         rewards = np.zeros(len(self.worms))
    #         for i in range(0,NB_TURNS_AFTER_TRAINING):
    #             rewards = self.step(rewards)
    #         print("DONE",rewards)
    #     brain.model.save(FilePath)
    #
    # def run(self):
    #     while not self.stop_signal:
    #         self.runEpisode()
    #
    # def stop(self):
    #     self.stop_signal = True
