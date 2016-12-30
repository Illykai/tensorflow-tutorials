"""
Unit tests for TicTacToe environment
"""
import unittest
from learning.environments import tic_tac_toe

class TestTicTacToeGame(unittest.TestCase):
    """
    Tests for TicTacToeGame
    """

    def test_set_state(self):
        """
        set_state test
        """
        ttt = tic_tac_toe.TicTacToeGame()
        test_state = [0, 0, 0,
                      0, 1, 2,
                      0, 0, 0]
        ttt.set_state(test_state)
        self.assertTrue(ttt.get_state() == ttt.get_state())

    def test_get_winner_none(self):
        """
        get_winner
        """
        ttt = tic_tac_toe.TicTacToeGame()
        test_state = [2, 1, 2,
                      1, 1, 2,
                      2, 2, 1]
        ttt.set_state(test_state)
        self.assertTrue(ttt.get_winner() == 0, "Expected %d, found %d" % (0, ttt.get_winner()))

    def test_get_winner_horizontal(self):
        """
        get_winner
        """
        ttt = tic_tac_toe.TicTacToeGame()
        test_state = [0, 0, 0,
                      1, 1, 1,
                      0, 0, 0]
        ttt.set_state(test_state)
        self.assertTrue(ttt.get_winner() == 1)

    def test_get_winner_vertical(self):
        """
        get_winner
        """
        ttt = tic_tac_toe.TicTacToeGame()
        test_state = [0, 1, 0,
                      0, 1, 0,
                      0, 1, 0]
        ttt.set_state(test_state)
        self.assertTrue(ttt.get_winner() == 1)

    def test_get_winner_diagonal_tr_bl(self):
        """
        get_winner
        """
        ttt = tic_tac_toe.TicTacToeGame()
        test_state = [0, 0, 2,
                      0, 2, 0,
                      2, 0, 0]
        ttt.set_state(test_state)
        self.assertTrue(ttt.get_winner() == 2)

    def test_get_winner_diagonal_tl_br(self):
        """
        get_winner
        """
        ttt = tic_tac_toe.TicTacToeGame()
        test_state = [2, 0, 0,
                      0, 2, 0,
                      0, 0, 2]
        ttt.set_state(test_state)
        self.assertTrue(ttt.get_winner() == 2)

def test_negamax_search():
    game = TicTacToeGame()
    # Handy test state for debugging
    game.set_state([0, 1, 0,
                    0, 0, 2,
                    0, 0, 0])
    action = get_negamax_action(game)
    print("Game state:")
    print(game)
    print("negamax action: %d" % action)


if __name__ == "__main__":
    unittest.main()
