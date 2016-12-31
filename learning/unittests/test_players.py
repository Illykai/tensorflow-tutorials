"""
Unit tests for the players module
"""
import unittest
from learning.environments.tic_tac_toe import TicTacToeGame
from learning.players.search import get_negamax_action
from learning.players.neural import generate_win_prediction_player_from_dummy

class TestPlayers(unittest.TestCase):
    """
    Test for negamax search
    """

    def test_negamax_search(self):
        """Test negamax search"""
        game = TicTacToeGame()
        # Handy test state for debugging
        game.set_state([0, 1, 0,
                        0, 0, 2,
                        0, 0, 0])
        action = get_negamax_action(game)
        self.assertTrue(action == 2 or action == 4)

    def test_dummy_win_prediction_player(self):
        """Test the dummy player"""
        game = TicTacToeGame()
        dummy = generate_win_prediction_player_from_dummy()
        action = dummy.get_action(game)
        self.assertTrue(action == 1, "Expected 1, got %d" % action)


if __name__ == "__main__":
    unittest.main()
