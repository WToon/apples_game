import random, itertools
from decision_logic import dqn


class AG_training_environment():

    def __init__(self, nb_apples=13, nb_agents=5):
        self.apples = self._init_apples(nb_apples)                  # list of apples in the game
        self.worms = self._init_worms(nb_agents)                    # list of worms in the game
        self.agents = [dqn.DQNAgent() for i in range(0,nb_agents)]  # list of agents (map to worms with index)

    @staticmethod
    def _init_apples(self, nb_apples):
        apples = [[random.randint(1,36), random.randint(1,16)] for i in range(0,nb_apples)]
        apples.sort()
        apples = [k for k,_ in itertools.groupby(apples)]
        return apples

    @staticmethod
    def _init_worms(self, nb_agents):
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


    # ----------------------------------------------------------------------------
    # a worm does a certain action
    def worm_step(self,worm,actionid):
        self.steps +=1
        print (self.steps)
        action = ACTIONS[actionid]
        x,y,orientation = worm.state

        apples = worm.visibleApples
        newX=None
        newY=None
        newOrientation=None
        orientation = ORIENTATION[orientation]
        reward = 0
        if action == "move":
            newOrientation=ORIENTATION.index(orientation)
            if orientation=="up":
                newX=x
                newY = (y-1)%16
            elif orientation =="left":
                newX = (x-1)%36
                newY=y
            elif orientation == "right":
                newX = (x+1)%36
                newY=y
            else:
                newX = x
                newY = (y+1)%16

        elif action == "left":
            if orientation=="up":
                newOrientation=ORIENTATION.index("left")
                newX=(x-1)%36
                newY = y
            elif orientation =="left":
                newOrientation=ORIENTATION.index("down")
                newX = x
                newY=(y+1)%16
            elif orientation == "right":
                newOrientation=ORIENTATION.index("up")
                newX = x
                newY=(y-1)%16
            else:
                newOrientation=ORIENTATION.index("right")
                newX = (x+1)%36
                newY = y
        else:
            if orientation=="up":
                newOrientation=ORIENTATION.index("right")
                newX=(x+1)%36
                newY = y
            elif orientation =="left":
                newOrientation=ORIENTATION.index("up")
                newX = x
                newY=(y-1)%16
            elif orientation == "right":
                newOrientation=ORIENTATION.index("down")
                newX = x
                newY=(y+1)%16
            else:
                newOrientation=ORIENTATION.index("left")
                newX = (x-1)%36
                newY = y
        if [newX, newY] in apples:
            reward +=1
            self.apples.remove([newX,newY])
            newApple = [random.randint(0,35),random.randint(0,36)]
            while newApple in self.apples:
                newApple = [random.randint(0, 35), random.randint(0, 36)]
            self.apples.append(newApple)
        newApples = self.checkApples(newX,newY,self.apples)
        return  ((newX,newY,newOrientation),newApples,reward)

    #check which apples are visible for the agent
    def checkApples(self,x,y,apples):
        result = []
        for i in apples:
            dx = abs(i[0]-x)
            if dx>18:
                dx = 36-dx
            dy = abs(i[1]-y)
            if dy>8:
                dy = 16-dy
            if (dx<=7):
                if (dy<=7):
                    result.append(i)
        return result

    #add apples to nearby positions (same as in the game)
    def addApples(self):
        for x in range(0,36):
            for y in range(0,16):
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
                    newX = (x+nbx)%36
                    newY = (x+nby)%16
                    if self.distance([x, y], [newX, newY]) <= 2 and [newX,newY] in self.apples:
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

    #reset the worm (new position etc)
    def wormreset(self,worm,apples):
        randomX = random.randint(0,35)
        randomY = random.randint(0,15)
        orientation=random.randint(0,3)
        worm.state = (randomX,randomY,orientation)
        worm.visibleApples = self.checkApples(randomX,randomY,apples)

    #reset the environment
    def reset(self):

        self.apples = []
        for i in range(0, NB_APPLES):
            newApple = [random.randint(0, 35), random.randint(0, 15)]
            while newApple in self.apples:
                newApple = [random.randint(0, 35), random.randint(0, 15)]
            self.apples.append(newApple)
        for worm in self.worms:
            self.wormreset(worm,self.apples)

    #take a step in the environment -> all worms do a move
    def step(self,rewards):
        for worm in self.worms:
            x,y,orientation = worm.state
            worm.state = (x,y,orientation)
            worm.visibleApples = self.checkApples(x,y,self.apples)
            action = worm.act()
            newState,apples,reward = self.worm_step(worm,action)
            worm.train(worm.state,action,reward,newState)
            worm.state = newState
            worm.visibleApples = apples
            rewards[self.worms.index(worm)]+= reward

        # self.addApples()
        return rewards

    #play one game
    def runEpisode(self):
        self.reset()
        rewards = np.zeros(len(self.worms))
        for i in range(0,NB_TURNS):
            rewards = self.step(rewards)
        self.steps = 0
        print("DONE",rewards)

    #play a number of games after training
    def runEpisodeAfterTraining(self):
        for x in range(0,EPISODES_AFTER_TRAINING):
            self.reset()
            rewards = np.zeros(len(self.worms))
            for i in range(0,NB_TURNS_AFTER_TRAINING):
                rewards = self.step(rewards)
            print("DONE",rewards)
        brain.model.save(FilePath)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True
