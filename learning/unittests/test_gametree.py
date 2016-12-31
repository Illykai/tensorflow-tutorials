"""
Unit tests for gametree module
"""
import unittest
from learning.environments.tic_tac_toe import TicTacToeGame
from learning.players.search import get_negamax_action

class TestTicTacToeGame(unittest.TestCase):
    """
    Tests for TicTacToeGame
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

if __name__ == "__main__":
    unittest.main()
