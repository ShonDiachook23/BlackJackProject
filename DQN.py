import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from BlackJackEnv import BlackjackEnv, Action  # ייבוא סביבת המשחק והפעולות האפשריות

# סוכן מבוסס Deep Q-Network
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # גודל הווקטור שמתאר את מצב הסביבה (state)
        self.action_size = action_size  # מספר הפעולות האפשריות (Hit, Stand)

        # זיכרון לשמירת חוויות קודמות – לצורך Replay
        self.memory = deque(maxlen=10000)

        # היפר-פרמטרים ללמידה
        self.gamma = 0.95             # פקטור הנחה – קובע עד כמה מתחשבים בתגמולים עתידיים
        self.epsilon = 1.0            # הסתברות לבחירה אקראית של פעולה (exploration)
        self.epsilon_min = 0.01       # ערך מינימלי ל-epsilon
        self.epsilon_decay = 0.995    # קצב דעיכה של epsilon בכל אפיזודה

        self.model = self.build_model()  # בניית הרשת הנוירונית

    # בניית רשת נוירונים פשוטה עם שתי שכבות חבויות
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))  # שכבת קלט + ReLU
        model.add(Dense(24, activation='relu'))                             # שכבת חבויה נוספת
        model.add(Dense(self.action_size, activation='linear'))            # שכבת פלט – ערך Q לכל פעולה
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))     # אימון עם פונקציית הפסד MSE
        return model

    # שמירת חוויה בזיכרון: מצב, פעולה, תגמול, מצב הבא, האם הסתיים
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # בחירת פעולה – או לפי הרשת (exploit) או באקראי (explore)
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(list(Action))  # פעולה אקראית (חקר)
        act_values = self.model.predict(np.array([state]), verbose=0)
        return Action(np.argmax(act_values[0]))  # פעולה עם ערך Q הגבוה ביותר

    # למידה מהזיכרון: replay של דוגמאות קודמות
    def replay(self, batch_size=32):
        # דגימה אקראית של מיני־באץ'
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # אם המשחק לא הסתיים – הוספת ערך Q עתידי (לפי מקסימום פעולות עתידיות)
                target += self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])

            # חיזוי ערכי Q נוכחיים ועדכון הפעולה שבוצעה לפי ה-target
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action.value] = target

            # התאמת הרשת לדוגמה החדשה
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

        # עדכון epsilon – צמצום רמת החקר עם הזמן
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# בלוק הפעלה – אימון הסוכן
if __name__ == '__main__':
    env = BlackjackEnv()  # יצירת סביבה
    agent = DQNAgent(state_size=3, action_size=len(Action))  # יצירת סוכן

    episodes = 1000  # מספר אפיזודות אימון
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

        # הדפסה בכל אפיזודה
        print(f"[{e+1}/{episodes}] Epsilon: {agent.epsilon:.4f} | Episode Reward: {episode_reward:.2f} | Total Balance: {total_balance:.2f}")

        # שמירה כל 500 אפיזודות
        if (e + 1) % 500 == 0:
            agent.model.save("blackjack_model.keras")
            print("🟢 Intermediate model saved (blackjack_model.keras)")

# שמירה סופית
agent.model.save("blackjack_model.keras")
print("✅ Final model saved: blackjack_model.keras")

agent.model.save("dqn_blackjack_model.h5")
print("✅ Final model saved: dqn_blackjack_model.h5")
