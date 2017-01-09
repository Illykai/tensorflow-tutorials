"""
Game running framework thing
"""
import datetime
import pickle
from learning.environments.tic_tac_toe import TicTacToeGame
from learning.players.basic import RandomPlayer
from learning.players import search

DATA_DIR = "data"

def main():
    """Do the cool things"""
    num_games = 1000
    game = TicTacToeGame()
    players = []
    players.append(RandomPlayer())
    players.append(RandomPlayer())
    runner = GameRunner(game, players)
    runner.run_tournament(num_games)

    # generate_random_game_data()
    # generate_negamax_game_data()
    # generate_tic_tac_toe_game_tree_pickle_file()

def generate_game_data(game, players, num_games, file_suffix):
    """Generates data from bots playing a game against each other

    Args:
        game: The Game to be played
        players: A list of Players for the game
        num_games: The number of rounds to players
        file_suffix: The file suffix for the generated data
    """
    # Data file setup
    date_now = datetime.datetime.now()
    date_string = date_now.strftime("%Y_%m_%d_%H_%M_%S")
    filename = "%s_games_%d_%s.csv" % (date_string, num_games, file_suffix)
    out_file = open("%s/%s" % (DATA_DIR, filename), "w")

    game_runner = GameRunner(game, players)
    for count in range(num_games):
        states, actions, winner = game_runner.run_game()

        if count % 100 == 0:
            print("\n[Run %d/%d]" % (count, num_games))
            print("Final state:")
            print(game.state_to_string(states[-1], 0))
        else:
            print(".", end="", flush=True)

        state_strings = [str(state) for state in states]
        action_strings = [str(action) for action in actions]
        # Because state vectors are comma separated numbers we need to put
        # them in  quotes for csv format
        out_file.write("\"" + "\",\"".join(state_strings) + "\"" + "\n")
        out_file.write(",".join(action_strings) + "\n")
        out_file.write(str(winner) + "\n")

def generate_tic_tac_toe_game_tree_pickle_file():
    """Generate a game tree pickle file for TicTacToe"""
    game = TicTacToeGame()
    pickle_filename = "%s/tic_tac_toe_game_tree.pkl" % DATA_DIR
    pickle_file = open(pickle_filename, "wb")
    game_tree = search.generate_game_tree(game, game.get_state())
    pickle.dump(game_tree, pickle_file)
    pickle_file.close()

def generate_negamax_game_data():
    """Generate a bunch of data from an optimal negamax player beating on a random player"""
    suffix = "_tic_tac_toe_negamax_vs_random_games"
    num_games = 10000
    game = TicTacToeGame()
    players = []
    pickle_filename = "%s/tic_tac_toe_game_tree.pkl" % DATA_DIR
    negamax_player = search.generate_player_from_pickle_file(pickle_filename)
    players.append(negamax_player)
    players.append(RandomPlayer())
    generate_game_data(game, players, num_games, suffix)

def generate_random_game_data():
    """Generate a bunch of data from random game players"""
    suffix = "_tic_tac_toe_random_vs_random_games"
    num_games = 1000
    game = TicTacToeGame()
    players = []
    players.append(RandomPlayer())
    players.append(RandomPlayer())
    generate_game_data(game, players, num_games, suffix)

class GameRunner:
    """
    Class for coordinating the running of a game
    """

    def __init__(self, game, players):
        """
        Args:
            game: The Game to be played
            players: A list of Players
        """
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

        return (state_history, action_history, self.game.get_winner())

    def run_tournament(self, num_games):
        """
        Run a num_games round tournament
        """
        date_now = datetime.datetime.now()
        date_string = date_now.strftime("%Y_%m_%d_%H_%M_%S")
        player_names = [player.get_name() for player in self.players]
        names_string = "_vs_".join(player_names)
        filename = "%s_%s_games_%d.csv" % (date_string, names_string, num_games)
        with open("%s/%s" % (DATA_DIR, filename), "w") as out_file:
            wins = [0] * (len(self.players) + 1)
            for _ in range(num_games):
                _, _, winner = self.run_game()
                wins[winner] += 1
            player_names = ["draw"] + player_names
            out_file.write(",".join(player_names) + "\n")
            wins_string = [str(num) for num in wins]
            out_file.write(",".join(wins_string) + "\n")
            percents_string = [str(num/num_games) for num in wins]
            out_file.write(",".join(percents_string) + "\n")

if __name__ == "__main__":
    main()
