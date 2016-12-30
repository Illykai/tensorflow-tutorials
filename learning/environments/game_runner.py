"""
Game running framework thing
"""
import random
import datetime
from learning.environments.tic_tac_toe import TicTacToeGame
from learning.gametree.gametree import *

DATA_DIR = "data"

def main():
    """Do the cool things"""
    # generate_random_game_data()
    generate_negamax_game_data()

def generate_negamax_game_data():
    """
    Generate a bunch of data from random game players
    """
    suffix = "_tic_tac_toe_negamax_vs_random_games_%d"
    # Run games
    game = TicTacToeGame()
    players = []
    game_tree = compute_game_tree(game, game.get_state())
    players.append(NegamaxPlayer(game_tree))
    players.append(RandomPlayer())
    generate_game_data(game, players, suffix)

def generate_random_game_data():
    """
    Generate a bunch of data from random game players
    """
    suffix = "_tic_tac_toe_random_vs_random_games_%d"
    # Run games
    game = TicTacToeGame()
    players = []
    players.append(RandomPlayer())
    players.append(RandomPlayer())
    generate_game_data(game, players, suffix)

def generate_game_data(game, players, file_suffix):
    # Test params
    num_games = 10
    date_now = datetime.datetime.now()
    date_string = date_now.strftime("%Y_%m_%d_%H_%M_%S")
    filename = "%s_games_%d_%s.csv" % (date_string, num_games, file_suffix)
    out_file = open(DATA_DIR + "/" + filename, "w")

    game_runner = GameRunner(game, players)
    for _ in range(num_games):
        states, actions, winner = game_runner.run_game()
        state_strings = [str(state) for state in states]
        action_strings = [str(action) for action in actions]
        # Because state vectors are comma separated numbers we need to put
        # them in  quotes for csv format
        out_file.write("\"" + "\",\"".join(state_strings) + "\"" + "\n")
        out_file.write(",".join(action_strings) + "\n")
        out_file.write(str(winner) + "\n")

class GameRunner:
    """
    Class for running and tracking the stats of games
    """

    def __init__(self, game, players):
        self.game = game
        self.players = players

    def run_game(self):
        """
        Start the game with the provided players
        """
        self.game.reset()
        player_turn = 0

        state_history = [self.game.get_state()]
        action_history = []

        while not self.game.is_over():
            action = self.players[player_turn].get_action(self.game)
            action_history.append(action)
            self.game.move(action)
            state_history.append(self.game.get_state())
            player_turn = player_turn + 1
            player_turn = player_turn % len(self.players)

        print("Final state")
        print(self.game)

        return (state_history, action_history, self.game.get_winner())

class Player:
    """
    Generic player interface
    """

    def get_action(self, game):
        """
        Get the player's move given the current game state
        """
        return 0

class HumanPlayer(Player):
    """
    A human player
    """

    def get_action(self, game):
        valid_moves = game.get_valid_moves()
        valid_moves = [str(move) for move in valid_moves]
        valid_moves.append("q")
        command = ""
        while command not in valid_moves:
            print(game)
            prompt = "Choose a move " + ", ".join(valid_moves) + ": "
            command = input(prompt)
            if command == "q":
                quit()
        move = int(command)
        return move

class RandomPlayer(Player):
    """
    A random player
    """

    def get_action(self, game):
        return random.choice(game.get_valid_moves())

class NegamaxPlayer(Player):
    """
    Optimal player that uses the minimax algorithm
    """

    def __init__(self, game_tree):
        """
        Set up a minimax Player

        Args:
            game_tree: The game tree for the game we're playing
        """
        self.game_tree = game_tree
        # Build a big dictionary for lookup into our tree
        self.state_to_node_dict = {}
        stack = []
        stack.append(game_tree)
        while stack:
            # Using the booleanity of pythonic lists
            node = stack.pop()
            self.state_to_node_dict[str(node.state)] = node
            for _, child in node.children.items():
                stack.append(child)

    def get_action(self, game):
        state = game.get_state()
        node = self.state_to_node_dict[str(state)]
        action, _ = negamax_search(game, node, True)
        game.set_state(state)
        return action

if __name__ == "__main__":
    main()
