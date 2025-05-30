import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from BlackJackEnv import BlackjackEnv, Action  # ייבוא הסביבה והפעולות

# סוכן מבוסס Deep Q-Network
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.build_model()
        self.losses = []  # לשמירת ערכי ה-loss לצורך גרף

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(list(Action))
        act_values = self.model.predict(np.array([state]), verbose=0)
        return Action(np.argmax(act_values[0]))

    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        batch_losses = []

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])

            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action.value] = target

            history = self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
            batch_losses.append(history.history['loss'][0])

        if batch_losses:
            self.losses.append(np.mean(batch_losses))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# הפעלת הסוכן והאימון
if __name__ == '__main__':
    env = BlackjackEnv()
    agent = DQNAgent(state_size=3, action_size=len(Action))

    episodes = 1000
    total_balance = 0

    for e in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_balance += reward

        agent.replay()

        print(f"[{e+1}/{episodes}] Epsilon: {agent.epsilon:.4f} | Episode Reward: {episode_reward:.2f} | Total Balance: {total_balance:.2f}")

        if (e + 1) % 500 == 0:
            agent.model.save("blackjack_model.keras")
            print("🟢 Intermediate model saved (blackjack_model.keras)")

    # שמירה סופית
    agent.model.save("blackjack_model.keras")
    print("✅ Final model saved: blackjack_model.keras")

    agent.model.save("dqn_blackjack_model.h5")
    print("✅ Final model saved: dqn_blackjack_model.h5")
