"""
Players that use neural networks to choose actions
"""

import csv
import numpy
import random
from learning.players.basic import Player

def main():
    """Handy entry point for testing"""
    load_training_data()

def generate_win_prediction_player_from_dummy():
    """Make a dummy player for testing"""
    dummy_net = DummyNetwork()
    return WinPredictionPlayer(dummy_net)

class WinPredictionPlayer(Player):
    """
    Player that plays by looking ahead and picking the action their NN
    thinks will win
    """

    def __init__(self, network):
        self.network = network

    def get_action(self, game):
        """
        Get the player's move given the current game state
        """

        valid_actions = game.get_valid_moves()
        current_state = game.get_state()
        successors = [(game.get_state_successor(current_state, action), action)
                      for action in valid_actions]
        player = game.get_active_player()
        best_prob = -1.0
        best_actions = []
        for (state, action) in successors:
            win_probs = self.network.query(state)
            if win_probs[player] > best_prob:
                best_prob = win_probs[player]
                best_actions = [action]
            elif win_probs[player] == best_prob:
                best_actions.append(action)
        return random.choice(best_actions)

class DummyNetwork:
    """"
    Just a stub network for testing
    """

    def query(self, state):
        """Dummy query for testing"""
        if str(state) == str([0, 1, 0, 0, 0, 0, 0, 0, 0]):
            return [0.1, 0.7, 0.2]
        else:
            return [0.1, 0.3, 0.6]

# Data wrangling
def load_training_data():
    # Hard coding filenames for now
    states = []
    winners = []
    with open("data/2016_12_28_21_25_00_tic_tac_toe_random_vs_random_games_1000.csv",
              "r") as random_game_file:
        reader = csv.reader(random_game_file, delimiter=",", quotechar='"')
        count = 0
        history = []
        winner = []
        for row in reader:
            if count == 0:
                history = row
            elif count == 1:
                # This is a final state. Ignore it - should already be in history.
                pass
            else:
                winner = int(row[0])
                one_hot_winner_array = numpy.zeros(3, dtype=numpy.uint8)
                one_hot_winner_array[winner] = 1
                for state in history:
                    state = state[1:-1]
                    numbers = state.split(", ")
                    numbers = [int(num) for num in numbers]
                    state_array = numpy.array(numbers, dtype=numpy.uint8)
                    states.append(state_array)
                    winners.append(one_hot_winner_array)
            count = (count + 1) % 3
        state_array = numpy.array(states)
        winner_array = numpy.array(winners)
        print("State array - Shape %s - Type %s "
              % (str(state_array.shape), str(state_array.dtype)))
        print("Winner array - Shape %s - Type %s "
              % (str(winner_array.shape), str(winner_array.dtype)))
        return (state_array, winner_array)

if __name__ == "__main__":
    main()

