import numpy as np
import random
from enum import Enum

# הגדרת הצורות האפשריות של הקלפים (לבבות, תלתן וכו')
SUITS  = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

# הגדרת הערכים האפשריים לקלפים – כולל קלפים מספריים וקלפים תמונה
VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# פעולות אפשריות שהשחקן/סוכן יכול לבצע
class Action(Enum):
    STAND = 0
    HIT = 1

# מחלקה לייצוג קלף בודד
class Card:
    def __init__(self, suit, value):
        self.suit = suit  # הצורה של הקלף (Spades, Hearts וכו')
        self.value = value  # הערך של הקלף (למשל 'K', '10', 'A')

    def get_numeric_value(self):
        # ממיר ערכים לא מספריים למספרים: J/Q/K שווים 10, A שווה 1 (שימושי בהמשך)
        if self.value in ['J', 'Q', 'K']:
            return 10
        elif self.value == 'A':
            return 1
        return int(self.value)

# מחלקה לניהול חפיסת הקלפים (כוללת ערבוב ומשיכה)
class Deck:
    def __init__(self, num_decks=6):
        self.num_decks = num_decks
        self.build()

    def build(self):
        # בונה את החפיסה מחדש עם מספר חפיסות (למשחק ריאליסטי)
        self.cards = [Card(suit, value) for suit in SUITS for value in VALUES] * self.num_decks
        random.shuffle(self.cards)

    def draw(self):
        # אם נותרו פחות מ-25% מהקלפים – מערבבים מחדש
        if len(self.cards) < (0.25 * 52 * self.num_decks):
            self.build()
        return self.cards.pop()  # משיכת קלף מהחפיסה

# מחלקה לייצוג יד של שחקן/דילר
class Hand:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)  # הוספת קלף ליד

    def get_value(self):
        # מחשבת את ערך היד הכוללת, תוך טיפול באסים
        value = sum(card.get_numeric_value() for card in self.cards)
        aces = sum(1 for card in self.cards if card.value == 'A')

        # אם יש אס/ים – אפשר להפוך אחד מהם ל-11 כל עוד לא עוברים את 21
        while aces > 0 and value + 10 <= 21:
            value += 10
            aces -= 1

        return value

# מחלקת הסביבה המרכזית של המשחק – מותאמת ללמידת חיזוק
class BlackjackEnv:
    def __init__(self, num_decks=6):
        self.deck = Deck(num_decks)
        self.reset()

    def reset(self):
        # מאתחל יד חדשה לשחקן ולדילר (כולל חלוקת קלפים התחלתית)
        self.player_hand = Hand()
        self.dealer_hand = Hand()

        # חלוקת שני קלפים לשחקן ואחד גלוי לדילר
        self.player_hand.add_card(self.deck.draw())
        self.player_hand.add_card(self.deck.draw())
        self.dealer_hand.add_card(self.deck.draw())

        self.done = False  # המשחק טרם הסתיים
        return self.get_state()  # מחזיר את מצב הפתיחה

    def step(self, action):
        # הסוכן מבצע פעולה: HIT או STAND
        if action == Action.HIT:
            self.player_hand.add_card(self.deck.draw())  # מוסיפים קלף לשחקן
            if self.player_hand.get_value() > 21:
                self.done = True  # השחקן הפסיד (עבר את 21)
                return self.get_state(), -1, self.done
            return self.get_state(), 0, self.done  # ממשיכים לשחק

        elif action == Action.STAND:
            self.done = True
            # תור הדילר – חושף קלף נוסף ומשחק עד שהוא מגיע ל-17 לפחות
            self.dealer_hand.add_card(self.deck.draw())
            while self.dealer_hand.get_value() < 17:
                self.dealer_hand.add_card(self.deck.draw())

            # השוואה בין ערך הידיים של השחקן והדילר לקביעת התוצאה
            player_val = self.player_hand.get_value()
            dealer_val = self.dealer_hand.get_value()

            if dealer_val > 21 or player_val > dealer_val:
                reward = 1  # ניצחון
            elif player_val == dealer_val:
                reward = 0  # תיקו
            else:
                reward = -1  # הפסד

            return self.get_state(), reward, self.done

    def get_state(self):
        # מחזיר ייצוג מופשט של המצב – כדי שסוכן למידת חיזוק יוכל לפעול עליו
        return (
            self.player_hand.get_value(),  # סכום היד של השחקן
            self.dealer_hand.cards[0].get_numeric_value(),  # ערך קלף הדילר הגלוי
            int(any(card.value == 'A' for card in self.player_hand.cards)),  # האם יש אס שמיש
        )
    
    def get_full_hands(self):
        # מחזיר את כל הקלפים של השחקן והדילר – לצורך הצגה בממשק הגרפי
        return {
            'player': [(card.value, card.suit) for card in self.player_hand.cards],
            'dealer': [(card.value, card.suit) for card in self.dealer_hand.cards]
        }
