import os
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.models import Sequential
from decision_logic import oh_encoding as ohe


#array of the actions that can be played
ACTIONS = ["move","left","right","fire"]
# Position, orientation, apples, other players (orientation/location)
state_size = 15*15*6

# move, left, right, fire
action_size = 4

# Hyper-paramater (power of 2)
batch_size = 32

# How many games played for training
n_episodes = 1001

output_dir = 'model_output/dqn_agent'
if not os.path.exists('model_output/dqn_agent'):
    os.makedirs(output_dir)


class DQNAgent():
    def __init__(self):
        self.state_size = state_size
        self.action_size = action_size
        # Memory (sample learning data from game)
        self.memory = deque(maxlen=2000)
        # Discount factor
        self.gamma = 0.95
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
        self.previous_reward=None
        self.previous_action=None

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(64, (2,2), input_shape=(6, 15, 15)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma*np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def next_action(self,player,players,apples):
      self.state = np.reshape(ohe.encode_state(player,players,apples), [1, 6,15,15])
      #now we know the next state, we can train the model
      if len(self.previous_state) !=0:
        self.remember(self.previous_state, self.previous_action, self.previous_reward, self.state, False) #train based on the previous action
      if len(self.memory) > batch_size:
        self.replay(batch_size)  # train the agent by replaying the experiences of the episode
      action = self.act(self.state)
      while action not in [0,1,2,3]:
        action=self.act(self.state)
      self.previous_reward = self.get_reward_after_action(player,players,apples,ACTIONS[action])
      self.previous_action=action
      self.previous_state = self.state
      return ACTIONS[action]



    def get_reward_after_action(self,player,players,apples,action):
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

