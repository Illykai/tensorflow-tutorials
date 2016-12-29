"""
Game running framework thing
"""
import random
import datetime
from learning.environments.tic_tac_toe import TicTacToeGame

def main():
    """
    Entry point
    """
    # Test params
    num_games = 10
    data_dir = "data"
    date_now = datetime.datetime.now()
    date_string = date_now.strftime("%Y_%m_%d_%H_%M_%S")
    filename = "%s_tic_tac_toe_random_vs_random_games_%d.csv" % (date_string, num_games)
    out_file = open(data_dir + "/" + filename, "w")

    # Run games
    ttt_game = TicTacToeGame()
    players = []
    # players.append(HumanPlayer())
    players.append(RandomPlayer())
    players.append(RandomPlayer())
    game_runner = GameRunner(ttt_game, players)

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

    def __init__(self):
        pass

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

if __name__ == "__main__":
    main()
