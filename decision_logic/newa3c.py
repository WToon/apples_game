
import numpy as np
import tensorflow as tf
from decision_logic import oh_encoding as ohe
import  time, random, threading
from keras.models import *
from keras.layers import *
import os
output_dir = 'model_output/a3c_agent'
if not os.path.exists('model_output/a3c_agent'):
    os.makedirs(output_dir)
# -- constants
ACTIONS = ['move','left','right']
RUN_TIME = 60
THREADS = 4
OPTIMIZERS = 2
THREAD_DELAY = 0.001


GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient
NUM_ACTIONS=3
NUM_STATE = 15*15*6
NONE_STATE = np.zeros(NUM_STATE)

class Agent:

    def __init__(self,training=False,brain=True):
      self.eps_start = 0.95
      self.eps_end = 0.01
      self.eps_steps =  75
      self.memory = []  # used for n_step return
      self.state = None
      self.previous_state = []
      self.previous_reward = None
      self.previous_action = None
      self.optimized = not training
      self.R=0

      from decision_logic import a3cbrain as br
      self.brain=None
      if (brain == True):
        self.brain = br.Brain()
      else:
        self.brain=brain
      self.frames = 0

    def reset(self):
      self.R=0
      self.frames = 0
      self.memory = []
      # Discount factor
      self.gamma = 0
      # Exploration rate (randomly explore new actions)
      # Initially exploration >> exploitation
      self.epsilon = 1.0
      # With ongoing playing, we shift towards exploitation
      self.epsilon_decay = 0.995
      # Epsilon floor as to not stop exploring
      self.epsilon_min = 0.01
      self.learning_rate = 0.001
      self.state = None
      self.previous_state = []
      self.previous_reward = None
      self.previous_action = None

    def get_reward_after_action(self, player, players, apples, action):
      orientation = players[player - 1].get('orientation')
      x = players[player - 1].get('location')[0]
      y = players[player - 1].get('location')[1]
      reward = 0
      if action == "move":
        if orientation == "up":
          newx = x
          newy = (y - 1) % 16
        elif orientation == "left":
          newx = (x - 1) % 36
          newy = y
        elif orientation == "right":
          newx = (x + 1) % 36
          newy = y
        else:
          newx = x
          newy = (y + 1) % 16

      elif action == "left":
        if orientation == "up":
          newx = (x - 1) % 36
          newy = y
        elif orientation == "left":
          newx = x
          newy = (y + 1) % 16
        elif orientation == "right":
          newx = x
          newy = (y - 1) % 16
        else:
          newx = (x + 1) % 36
          newy = y
      elif action == "right":
        if orientation == "up":
          newx = (x + 1) % 36
          newy = y
        elif orientation == "left":
          newx = x
          newy = (y - 1) % 16
        elif orientation == "right":
          newx = x
          newy = (y + 1) % 16
        else:
          newx = (x - 1) % 36
          newy = y
      else:
        newx = x
        newy = y
        reward -= 1
      if newx == 0:
        newx = 36
      if newy == 0:
        newy = 16
      if [newx, newy] in apples:
        reward += 1
      return reward

    def next_action(self,player,players,apples,training):
      if (not self.optimized):
        self.optimized=True
        from decision_logic import game_env as ge
        from decision_logic import a3coptimizer as opt
        envs = [ge.AGTrainingEnvironment(13,2,False,self.brain) for i in range(THREADS)]
        opts = [opt.Optimizer(self.brain) for i in range(OPTIMIZERS)]

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
        self.save(output_dir + '/WeightsafterTraining')

      self.state = np.reshape(ohe.encode_state(player, players, apples), [1, 1350])
      # now we know the next state, we can train the model
      if training:
        if len(self.previous_state) != 0:
          self.train(self.previous_state, self.previous_action, self.previous_reward, self.state)  # train based on the previous action
      action = self.act(self.state)
      while action not in [0, 1, 2]:
        action = self.act(self.state)
      if training:
        self.previous_reward = self.get_reward_after_action(player, players, apples, ACTIONS[action])
        self.previous_action = action
        self.previous_state = self.state
      return ACTIONS[action]


    def getEpsilon(self):
      if (self.frames >= self.eps_steps):
        return self.eps_end
      else:
        return self.eps_start + self.frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s):
      eps = self.getEpsilon()
      self.frames+= 1

      if random.random() < eps:
        return random.randint(0, NUM_ACTIONS - 1)

      else:
        s = np.array([s])
        p = self.brain.predict_p(s)[0]
        # a = np.argmax(p)
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
          self.brain.train_push(s, a, r, s_)

          self.R = (self.R - self.memory[0][2]) / GAMMA
          self.memory.pop(0)

        self.R = 0
      if len(self.memory) >= N_STEP_RETURN:
        s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
        self.brain.train_push(s, a, r, s_)
        self.R = self.R - self.memory[0][2]
        self.memory.pop(0)


    def load(self, name=output_dir+'/Weights'):
        self.brain.model.load_weights(name)

    def save(self, name=output_dir+'/Weights'):
        self.brain.model.save_weights(name)










