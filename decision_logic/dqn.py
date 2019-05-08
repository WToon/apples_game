import os
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import plot_model


# Position, orientation, apples, other players (orientation/location)
state_size = 15*15*6

# move, left, right
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

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential(name="Q_Network")
        model.add(Conv2D(64, (2,2), input_shape=(6, 15, 15), name="States"))
        model.add(Dense(24, activation='relu', name="Extra_convergence"))
        model.add(Dense(self.action_size, activation='linear', name="Probability_distribution_for_the_4_actions"))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        plot_model(model, "model.png", show_shapes=True)

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

