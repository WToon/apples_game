import numpy as np
import tensorflow as tf

import gym, time, random, threading
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NB_APPLES = 50
NB_AGENTS = 5
NB_TURNS = 100
NB_TURNS_AFTER_TRAINING=1000
EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 750
GAMMA = 0.99
ACTIONS = ["left","move","right"]
ORIENTATION = ["up","left","right","down"]
N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN
NUM_ACTIONS = 3
NUM_STATE = 3
NONE_STATE = np.zeros(NUM_STATE)

RUN_TIME = 100
THREADS = 8
OPTIMIZERS = 2

FilePath = "C:/Users/siebe/OneDrive/Documenten/School/machine learning project/apples_game/decision_logic/trainedModel.h5"
THREAD_DELAY = 0.001
MIN_BATCH = 4*32
LEARNING_RATE = 5e-3

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient
frames = 0
class A3CAgent:
  def __init__(self, eps_start, eps_end, eps_steps, x, y, orientation, apples):
    self.eps_start = eps_start
    self.eps_end = eps_end
    self.eps_steps = eps_steps
    self.state = (x, y, orientation, apples)
    self.memory = []  # used for n_step return
    self.R = 0.
    self.visibleApples = apples
    self.brain = Brain()

  def getEpsilon(self):
    if (frames >= self.eps_steps):
      return self.eps_end
    else:
      return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

  def get_A3C_decision(self,player, players, apples):
    # Player position
    position = players[player - 1]["location"]
    # Player orientation
    orientation = players[player - 1]["orientation"]
    self.visibleApples=apples
    self.state = (position[0],position[1],ORIENTATION.index(orientation))
    actionid = self.act()
    #actionid defines the action that will be played, the followinig is to compute the reward
    newX = None
    newY = None
    newOrientation = None
    action = ACTIONS[actionid]
    x = position[0]
    y = position[1]
    reward = 0
    if action == "move":
      newOrientation = ORIENTATION.index(orientation)
      if orientation == "up":
        newX = x
        newY = (y - 1) % 16
      elif orientation == "left":
        newX = (x - 1) % 36
        newY = y
      elif orientation == "right":
        newX = (x + 1) % 36
        newY = y
      else:
        newX = x
        newY = (y + 1) % 16

    elif action == "left":
      if orientation == "up":
        newOrientation = ORIENTATION.index("left")
        newX = (x - 1) % 36
        newY = y
      elif orientation == "left":
        newOrientation = ORIENTATION.index("down")
        newX = x
        newY = (y + 1) % 16
      elif orientation == "right":
        newOrientation = ORIENTATION.index("up")
        newX = x
        newY = (y - 1) % 16
      else:
        newOrientation = ORIENTATION.index("right")
        newX = (x + 1) % 36
        newY = y
    else:
      if orientation == "up":
        newOrientation = ORIENTATION.index("right")
        newX = (x + 1) % 36
        newY = y
      elif orientation == "left":
        newOrientation = ORIENTATION.index("up")
        newX = x
        newY = (y - 1) % 16
      elif orientation == "right":
        newOrientation = ORIENTATION.index("down")
        newX = x
        newY = (y + 1) % 16
      else:
        newOrientation = ORIENTATION.index("left")
        newX = (x - 1) % 36
        newY = y

    if [newX, newY] in apples:
      reward += 1
    newState = (newX,newY,newOrientation)
    self.train(self.state, actionid, reward, newState)
    self.state = newState
    return action

  def act(self):
    s = self.state
    eps = self.getEpsilon()
    global frames;
    frames = frames + 1

    if random.random() < eps:
      return random.randint(0, NUM_ACTIONS - 1)

    else:
      s = np.array([s])
      p = self.brain.predict_p(s)[0]
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
  # ---------------------------------------------------------------------------------------


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

    self.default_graph.finalize()  # avoid modifications

  def _build_model(self):


    model = load_model(FilePath)
    model._make_predict_function()  # have to initialize before threading

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
      p, v = self.model.predict(s)
      return p

  def predict_v(self, s):
    with self.default_graph.as_default():
      p, v = self.model.predict(s)
      return v
