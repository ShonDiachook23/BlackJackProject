from flask import Flask, render_template, redirect, url_for, session, request
from BlackJackEnv import BlackjackEnv, Action
import tensorflow as tf
import numpy as np
import os
import time

app = Flask(__name__)
app.secret_key = os.urandom(24)

agent_model = tf.keras.models.load_model("blackjack_model.keras")
env = BlackjackEnv()

@app.route('/', methods=['GET', 'POST'])
def home():
    bet = int(request.form.get('bet', 10)) if request.method == 'POST' else 10
    state = env.reset()
    session['state'] = state
    session['bet'] = bet
    session['log'] = []
    session['done'] = False
    session['agent_play'] = False
    full_hands = env.get_full_hands()
    return render_template('game.html', state=state, full_hands=full_hands, bet=bet, log=[])

@app.route('/hit')
def hit():
    state, reward, done = env.step(Action.HIT)
    session['state'] = state
    session['done'] = done
    session['log'] = session.get('log', []) + [{'action': 'HIT', 'state': state, 'reward': reward, 'done': done}]
    return render_template('game.html', state=state, full_hands=env.get_full_hands(),
                           reward=reward, done=done, bet=session['bet'], log=session['log'])

@app.route('/stand')
def stand():
    state, reward, done = env.step(Action.STAND)
    session['state'] = state
    session['done'] = done
    session['log'] = session.get('log', []) + [{'action': 'STAND', 'state': state, 'reward': reward, 'done': done}]
    return render_template('game.html', state=state, full_hands=env.get_full_hands(),
                           reward=reward, done=done, bet=session['bet'], log=session['log'])

@app.route('/agent_move')
def agent_move():
    session['agent_play'] = True
    state = session.get('state')

    state_vector = np.array(state).reshape(1, -1)
    q_values = agent_model.predict(state_vector)[0]
    action = Action.HIT if np.argmax(q_values) == Action.HIT.value else Action.STAND

    state, reward, done = env.step(action)
    session['state'] = state
    session['done'] = done
    session['log'] = session.get('log', []) + [{'action': action.name, 'state': state, 'reward': reward, 'done': done}]

    return render_template('game.html', state=state, full_hands=env.get_full_hands(),
                           reward=reward, done=done, bet=session['bet'], log=session['log'], agent_play=True)

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)