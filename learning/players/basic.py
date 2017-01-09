"""
Basic player classes
"""

import random

class Player:
    """
    Generic player interface
    """

    def get_action(self, game):
        """
        Get the player's move given the current game state
        """
        return 0

    def get_name(self):
        """
        Get the player's name
        """
        return "player"

class HumanPlayer(Player):
    """
    A shell for humans to play that prompts for their actions
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

    def get_name(self):
        """
        Get the player's name
        """
        return "human_player"

class RandomPlayer(Player):
    """
    A random player
    """

    def get_action(self, game):
        return random.choice(game.get_valid_moves())

    def get_name(self):
        """
        Get the player's name
        """
        return "random_player"
