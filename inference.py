import tensorflow as tf
from BlackJackEnv import BlackjackEnv
import numpy as np

# פונקציה שמבצעת הערכת ביצועים למודל DQN על סביבה של בלאק ג'ק
def evaluate_model(model, env, num_episodes=5000):
    total_reward = 0  # מצטבר של כל התגמולים

    for episode in range(num_episodes):
        state = env.reset()  # אתחול מצב חדש
        done = False

        while not done:
            # המרה של המצב לווקטור קלט עבור המודל
            # שימי לב: הקוד הזה מניח מצב בגודל 6 (לא תואם לגרסה שלנו)
            # ייתכן שהוא מתייחס לגרסה ישנה של BlackjackEnv
            state_vector = np.array([state[0], state[0], state[1], state[2], state[3], state[4]])
            state_tensor = tf.convert_to_tensor([state_vector], dtype=tf.float32)

            # חיזוי פעולה באמצעות המודל – בוחרים את הפעולה עם ערך Q הגבוה ביותר
            action = tf.argmax(model(state_tensor), axis=1).numpy()[0]

            # ביצוע הפעולה בסביבה
            state, reward, done = env.step(action)
            total_reward += reward  # צבירת התגמול

    # חישוב תגמול ממוצע לכל אפיזודה
    average_reward = total_reward / num_episodes
    return average_reward

# קוד ראשי שמופעל כאשר מריצים את הקובץ
if __name__ == "__main__":
    # טעינת המודל המאומן מקובץ H5 (כולל התאמה לפונקציית הפסד)
    model_path = "dqn_blackjack_model.h5"
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
    )

    # יצירת מופע של סביבת המשחק
    env = BlackjackEnv()

    # הרצת הערכת ביצועים והצגת ממוצע התגמול
    avg_reward = evaluate_model(model, env)
    print(f"Average reward over 5000 episodes: {avg_reward}")
