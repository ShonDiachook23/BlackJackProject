from flask import Flask, render_template, redirect, url_for, session, request
from BlackJackEnv import BlackjackEnv, Action
import tensorflow as tf
import numpy as np
import os
import time

# יצירת מופע Flask להפעלת שרת אינטרנט
app = Flask(__name__)
app.secret_key = os.urandom(24)  # מפתח להצפנת session (נדרש ל-Flask)

# טעינת המודל המאומן מסוג DQN מקובץ keras
agent_model = tf.keras.models.load_model("blackjack_model.keras")

# יצירת מופע של סביבת המשחק
env = BlackjackEnv()

# עמוד הבית – התחלת משחק חדש
@app.route('/', methods=['GET', 'POST'])
def home():
    # קריאת סכום ההימור מתוך הטופס, או ברירת מחדל ל־10
    bet = int(request.form.get('bet', 10)) if request.method == 'POST' else 10

    # אתחול מצב המשחק דרך סביבת הבלאק ג'ק
    state = env.reset()

    # שמירת מצב ההתחלה ב-session (כדי לשמר אותו בין בקשות)
    session['state'] = state
    session['bet'] = bet
    session['log'] = []
    session['done'] = False
    session['agent_play'] = False

    # מקבל את הידיים של השחקן והדילר (לטובת התצוגה)
    full_hands = env.get_full_hands()

    # טוען את תבנית ה־HTML עם הנתונים
    return render_template('game.html', state=state, full_hands=full_hands, bet=bet, log=[])

# פעולה של השחקן: HIT – משיכת קלף נוסף
@app.route('/hit')
def hit():
    # ביצוע פעולה של HIT בסביבה
    state, reward, done = env.step(Action.HIT)

    # עדכון המצב הנוכחי
    session['state'] = state
    session['done'] = done

    # עדכון יומן הפעולות (log)
    session['log'] = session.get('log', []) + [{'action': 'HIT', 'state': state, 'reward': reward, 'done': done}]

    # הצגת תבנית המשחק לאחר הפעולה
    return render_template('game.html', state=state, full_hands=env.get_full_hands(),
                           reward=reward, done=done, bet=session['bet'], log=session['log'])

# פעולה של השחקן: STAND – סיום התור והעברת התור לדילר
@app.route('/stand')
def stand():
    # ביצוע פעולה של STAND בסביבה
    state, reward, done = env.step(Action.STAND)

    # עדכון session ויומן הפעולות
    session['state'] = state
    session['done'] = done
    session['log'] = session.get('log', []) + [{'action': 'STAND', 'state': state, 'reward': reward, 'done': done}]

    return render_template('game.html', state=state, full_hands=env.get_full_hands(),
                           reward=reward, done=done, bet=session['bet'], log=session['log'])

# פעולה של הסוכן המאומן (DQN) – בוחר פעולה בעצמו לפי המודל
@app.route('/agent_move')
def agent_move():
    session['agent_play'] = True  # דגל שמציין שהסוכן הוא זה שפועל
    state = session.get('state')

    # המרת המצב לווקטור קלט למודל
    state_vector = np.array(state).reshape(1, -1)

    # חיזוי ערכי Q בעזרת המודל המאומן
    q_values = agent_model.predict(state_vector)[0]

    # בחירת פעולה לפי ערך Q הגבוה ביותר
    action = Action.HIT if np.argmax(q_values) == Action.HIT.value else Action.STAND

    # ביצוע הפעולה שבחר הסוכן
    state, reward, done = env.step(action)

    # עדכון session ויומן הפעולות
    session['state'] = state
    session['done'] = done
    session['log'] = session.get('log', []) + [{'action': action.name, 'state': state, 'reward': reward, 'done': done}]

    # הצגת מצב המשחק לאחר פעולת הסוכן
    return render_template('game.html', state=state, full_hands=env.get_full_hands(),
                           reward=reward, done=done, bet=session['bet'], log=session['log'], agent_play=True)

# אתחול המשחק – ניקוי session וחזרה לעמוד הבית
@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('home'))

# הפעלת שרת Flask
if __name__ == '__main__':
    app.run(debug=True)
