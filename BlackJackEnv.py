import numpy as np
import random
from enum import Enum

SUITS  = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
VALUES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

class Action(Enum):
    STAND = 0
    HIT = 1

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def get_numeric_value(self):
        if self.value in ['J', 'Q', 'K']:
            return 10
        elif self.value == 'A':
            return 1
        return int(self.value)

class Deck:
    def __init__(self, num_decks=6):
        self.num_decks = num_decks
        self.build()

    def build(self):
        self.cards = [Card(suit, value) for suit in SUITS for value in VALUES] * self.num_decks
        random.shuffle(self.cards)

    def draw(self):
        if len(self.cards) < (0.25 * 52 * self.num_decks):
            self.build()
        return self.cards.pop()

class Hand:
    def __init__(self):
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def get_value(self):
        value = sum(card.get_numeric_value() for card in self.cards)
        aces = sum(1 for card in self.cards if card.value == 'A')

        while aces > 0 and value + 10 <= 21:
            value += 10
            aces -= 1

        return value

class BlackjackEnv:
    def __init__(self, num_decks=6):
        self.deck = Deck(num_decks)
        self.reset()

    def reset(self):
        self.player_hand = Hand()
        self.dealer_hand = Hand()

        # Deal initial cards
        self.player_hand.add_card(self.deck.draw())
        self.player_hand.add_card(self.deck.draw())
        self.dealer_hand.add_card(self.deck.draw())  # Dealer's visible card

        self.done = False
        return self.get_state()

    def step(self, action):
        if action == Action.HIT:
            self.player_hand.add_card(self.deck.draw())
            if self.player_hand.get_value() > 21:
                self.done = True
                return self.get_state(), -1, self.done
            return self.get_state(), 0, self.done

        elif action == Action.STAND:
            self.done = True
            # Dealer reveals hidden card and plays
            self.dealer_hand.add_card(self.deck.draw())
            while self.dealer_hand.get_value() < 17:
                self.dealer_hand.add_card(self.deck.draw())

            player_val = self.player_hand.get_value()
            dealer_val = self.dealer_hand.get_value()

            if dealer_val > 21 or player_val > dealer_val:
                reward = 1
            elif player_val == dealer_val:
                reward = 0
            else:
                reward = -1

            return self.get_state(), reward, self.done

    def get_state(self):
        return (
            self.player_hand.get_value(),
            self.dealer_hand.cards[0].get_numeric_value(),
            int(any(card.value == 'A' for card in self.player_hand.cards)),
        )
    
    def get_full_hands(self):
            return {
        'player': [(card.value, card.suit) for card in self.player_hand.cards],
        'dealer': [(card.value, card.suit) for card in self.dealer_hand.cards]
    }