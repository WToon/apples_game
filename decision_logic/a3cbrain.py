import numpy as np
import tensorflow as tf
import  time, random, threading
from keras.models import *
from keras.layers import *
from keras import backend as K

from keras.optimizers import Adam
import os
# -- constants
ENV = 'CartPole-v0'
ACTIONS = ['move','left','right']
RUN_TIME = 60
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001


GAMMA = 0.99

N_STEP_RETURN = 6
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
class Brain:
  train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
  lock_queue = threading.Lock()

  def __init__(self):
    self.session = tf.Session()
    K.set_session(self.session)
    K.manual_variable_initialization(True)

    self.model = self._build_model()
    if os.path.exists('model_output/a3c_agent/WeightsafterTraining2'):
        self.model.load_weights('model_output/a3c_agent/WeightsafterTraining2')
    self.graph = self._build_graph(self.model)

    self.session.run(tf.global_variables_initializer())
    self.default_graph = tf.get_default_graph()


  def _build_model(self):

    l_input = Input(batch_shape=(None, NUM_STATE))
    l_dense = Dense(16, activation='relu')(l_input)
    out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
    out_value = Dense(1, activation='linear')(l_dense)

    model = Model(inputs=[l_input], outputs=[out_actions, out_value])
    #model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))

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
    v=[]
    for s__ in s_:
      v.append(self.predict_v(s__))
    v = np.vstack(v)
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
      s = np.reshape(s,[1,1350])
      p, v = self.model.predict(s)
      return p, v

  def predict_p(self, s):
    with self.default_graph.as_default():
      s = np.reshape(s,[1,1350])
      p, v = self.model.predict(s)
      return p

  def predict_v(self, s):
    with self.default_graph.as_default():
      s = np.reshape(s,[1,1350])
      p, v = self.model.predict(s)
      return v

