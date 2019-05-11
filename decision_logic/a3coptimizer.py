import numpy as np
import tensorflow as tf
from decision_logic import oh_encoding as ohe
from decision_logic import game_env as ge
import  time, random, threading
from keras.models import *
from keras.layers import *
# -- constants
ENV = 'CartPole-v0'
ACTIONS = ['move','left','right']
RUN_TIME = 60
THREADS = 8
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

class Optimizer(threading.Thread):
  stop_signal = False

  def __init__(self,brain):
    threading.Thread.__init__(self)
    self.brain = brain

  def run(self):
    while not self.stop_signal:
      self.brain.optimize()

  def stop(self):
    self.stop_signal = True
