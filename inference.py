import tensorflow as tf
from BlackJackEnv import BlackjackEnv, Action
import numpy as np

# פונקציה שמבצעת הערכת ביצועים למודל DQN על סביבה של בלאק ג'ק
def evaluate_model(model, env, num_episodes=5000):
    total_reward = 0  # מצטבר של כל התגמולים

    for episode in range(num_episodes):
        state = env.reset()  # אתחול מצב חדש
        done = False

        while not done:
            # המרה של המצב לווקטור קלט עבור המודל
            state_vector = np.array([state[0], state[1], state[2]])
            state_tensor = tf.convert_to_tensor([state_vector], dtype=tf.float32)

            # חיזוי פעולה באמצעות המודל – בוחרים את הפעולה עם ערך Q הגבוה ביותר
            action_index = tf.argmax(model(state_tensor), axis=1).numpy()[0]
            action = Action(action_index)  # המרה למשתמש Enum
            state, reward, done = env.step(action)


            # ביצוע הפעולה בסביבה
            state, reward, done = env.step(action)
            total_reward += reward  # צבירת התגמול

    # חישוב תגמול ממוצע לכל אפיזודה
    average_reward = total_reward / num_episodes
    return average_reward

def evaluate_against_basic_strategy(model, num_samples=1000):
    basic_strategy_hard = {
        (5, d): 1 for d in range(2, 12)
    } | {
        (6, d): 1 for d in range(2, 12)
    } | {
        (7, d): 1 for d in range(2, 12)
    } | {
        (8, d): 1 for d in range(2, 12)
    } | {
        (9, d): 1 if d in [2, 7, 8, 9, 10, 11] else 1 for d in range(2, 12)
    } | {
        (10, d): 1 if d in [10, 11] else 1 for d in range(2, 12)
    } | {
        (11, d): 1 for d in range(2, 12)
    } | {
        (12, d): 1 if d in [2, 3, 7, 8, 9, 10, 11] else 0 for d in range(2, 12)
    } | {
        (13, d): 0 if d in [2, 3, 4, 5, 6] else 1 for d in range(2, 12)
    } | {
        (14, d): 0 if d in [2, 3, 4, 5, 6] else 1 for d in range(2, 12)
    } | {
        (15, d): 0 if d in [2, 3, 4, 5, 6] else 1 for d in range(2, 12)
    } | {
        (16, d): 0 if d in [2, 3, 4, 5, 6] else 1 for d in range(2, 12)
    } | {
        (17, d): 0 for d in range(2, 12)
    }

    matches = 0
    for _ in range(num_samples):
        player_sum = np.random.randint(5, 18)
        dealer_card = np.random.randint(2, 12)
        usable_ace = False
        state = (player_sum, dealer_card, usable_ace)

        if (player_sum, dealer_card) not in basic_strategy_hard:
            continue

        recommended_action = basic_strategy_hard[(player_sum, dealer_card)]
        state_vector = np.array([state[0], state[1], state[2]])
        state_tensor = tf.convert_to_tensor([state_vector], dtype=tf.float32)
        predicted_action = tf.argmax(model(state_tensor), axis=1).numpy()[0]

        if predicted_action == recommended_action:
            matches += 1

    accuracy = matches / num_samples * 100
    print("Evaluation completed...")
    print("Number of test samples:", num_samples)
    print("Correct decisions:", matches)
    print(f"Success rate: {accuracy:.2f}%")



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
    print("Comparison by Recommendation Map")
    evaluate_against_basic_strategy(model, num_samples=1000)

