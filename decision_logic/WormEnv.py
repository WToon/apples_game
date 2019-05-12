#every worm does its turn, then extra apples get spawned
import numpy as np
import tensorflow as tf

import time, random, threading
import math
import numpy as np
from keras.models import *
from keras.layers import *
from keras import backend as K
from keras import optimizers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#TRAINING VARIABLES
NB_APPLES = 50
NB_AGENTS = 1
NB_TURNS = 100
NB_TURNS_AFTER_TRAINING=1000
EPISODES_AFTER_TRAINING=0

#AGENT VARIABLES
EPS_START = 0.9
EPS_STOP = .1
EPS_STEPS = 50
GAMMA = 0.99
ACTIONS = ["move","left","right"]
ORIENTATION = ["up","left","right","down"]
N_STEP_RETURN = 4
GAMMA_N = GAMMA ** N_STEP_RETURN
NUM_ACTIONS = len(ACTIONS)
NUM_STATE = 3
NONE_STATE = np.zeros(NUM_STATE)

RUN_TIME = 0
THREADS = 1
OPTIMIZERS = 0

#Filepath to save and load the model
FilePath = "D:/Users/Wouter/Documents/Repos/apples_game/decision_logic/trainedModel.h5"
THREAD_DELAY = 0.001
MIN_BATCH = 4*32
LEARNING_RATE = 5e-3

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient


class WormsEnvironment(threading.Thread):
    stop_signal = False


    def __init__(self):
        threading.Thread.__init__(self)
        self.apples = []
        self.worms = []
        self.steps = 0
        self.orientations = ['left','right','up']

        #add the apples
        for i in range(0,NB_APPLES):
            newApple = [random.randint(0,35),random.randint(0,15)]
            while newApple in self.apples:
                newApple = [random.randint(0, 35), random.randint(0, 15)]
            self.apples.append(newApple)
        #add the worms on a random position
        for i in range(0,NB_AGENTS):
            randx = random.randint(0,35)
            randy = random.randint(0,15)
            self.worms.append(Agent(EPS_START,EPS_STOP,EPS_STEPS,randx,randy,random.randint(0,3), self.checkApples(randx,randy,self.apples)))


        #a worm does a certain action
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



#------------------------------------------------------------------------------------------------------------------
frames = 0
class Agent:
    def __init__(self, eps_start, eps_end, eps_steps,x,y,orientation,apples):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.state = (x,y,orientation,apples)
        self.memory = []  # used for n_step return
        self.R = 0.
        self.visibleApples = apples
        self.orientations = ['left','right','up']

    def getEpsilon(self):
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self):
        s = self.state
        eps = self.getEpsilon()
        global frames;
        frames = frames + 1

        if random.random() < eps: #explore
            return random.randint(0, NUM_ACTIONS - 1)

        else: #exploit
            sarr = np.array([s])
            p = brain.predict_p(sarr)[0]
            a = np.random.choice(NUM_ACTIONS, p=p)
            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]
            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)
#---------------------------------------------------------------------------------------


class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

       # self.default_graph.finalize()  # avoid modifications

    def _build_model(self):
        l_input = Input(batch_shape=(None, NUM_STATE))
        l_dense = Dense(16, activation='relu')(l_input)

        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


        #model = load_model(FilePath)
        #model._make_predict_function()  # have to initialize before threading
        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keepdims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1,
                                               keepdims=True)  # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:  # more thread could have passed without lock
                return  # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            #s = np.reshape(s,[1,2])
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v


class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True

#main
print("Lets begin")
envs = [WormsEnvironment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]
brain = Brain()
for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Training finished")
env = WormsEnvironment()
env.runEpisodeAfterTraining()

