"""
TicTacToe testing framework
"""

def main():
    """
    Entry point for TicTacToe framework
    """

    game = TicTacToeGame()
    game.set_state([0, 0, 0,
                    0, 1, 2,
                    0, 0, 0])
    print(game)

class TicTacToeGame:
    """
    Simple representation of a TicTacToe game
    """

    CELL_EMPTY = 0
    CELL_P1 = 1
    CELL_P2 = 2

    def __init__(self):
        """
        Default contructor
        """
        self.state = []
        self.turn = 0
        self.reset()

    def move(self, action):
        """
        Move in actionth square.
        """
        if self.state[action] != self.CELL_EMPTY:
            return False
        else:
            player = self.turn % 2 + 1
            self.state[action] = player
            return True

    def get_state(self):
        """
        Return the raw state of the board. Turn number is not a part of the state.
        """
        return self.state

    def get_valid_moves(self):
        """
        The valid moves are any empty cell
        """
        moves = filter(lambda x: self.state[x] == self.CELL_EMPTY, range(9))
        return moves

    def reset(self):
        """
        Reset the state
        """
        self.set_state([self.CELL_EMPTY] * 9)

    def set_state(self, state):
        """
        Set the state of the board
        """
        self.state = state
        # The turn number is the number of moves made
        self.turn = 0
        for cell in self.state:
            if cell is not self.CELL_EMPTY:
                self.turn = self.turn + 1

    def __repr__(self):
        """
        This isn't the most efficient way to do string concatenation, but at least it's clear
        """
        result = ""
        for (index, cell) in enumerate(self.state):
            if cell == self.CELL_EMPTY:
                result += str(index)
            elif cell == self.CELL_P1:
                result += "X"
            elif cell == self.CELL_P2:
                result += "O"
            # Include a line break at the end of each row
            if index % 3 == 2:
                result += "\n"
        # Convert to human friendly turn number
        result += "Turn: %d" % (self.turn + 1)
        return result

if __name__ == "__main__":
    main()
