import tensorflow as tf
from BlackJackEnv import BlackjackEnv
import numpy as np

def evaluate_model(model, env, num_episodes=5000):
    total_reward = 0

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # Convert state to a flat vector with 6 elements
            state_vector = np.array([state[0], state[0], state[1], state[2], state[3], state[4]])
            state_tensor = tf.convert_to_tensor([state_vector], dtype=tf.float32)
            action = tf.argmax(model(state_tensor), axis=1).numpy()[0]
            state, reward, done = env.step(action)
            total_reward += reward

    average_reward = total_reward / num_episodes
    return average_reward

if __name__ == "__main__":
    # Load your trained model
    model_path = "dqn_blackjack_model.h5"
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
    )

    # Initialize the BlackJack environment
    env = BlackjackEnv()

    # Evaluate the model
    avg_reward = evaluate_model(model, env)
    print(f"Average reward over 5000 episodes: {avg_reward}")