"""
TicTacToe framework
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
            # +1 because 0 is the empty cell code
            player = self.turn % 2 + 1
            self.state[action] = player
            self.turn = self.turn + 1
            return True

    def get_state(self):
        """
        Return a copy of the raw state of the board. Turn number is not a part of the state.
        """
        return list(self.state)

    def get_valid_moves(self):
        """
        The valid moves are any empty cell
        """
        return [move for move in range(9) if self.state[move] == self.CELL_EMPTY]

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

    def get_winner(self):
        """
        If there is a winner, returns the player number, otherwise returns 0
        """

        # Horizontal
        for row in range(3):
            row_start = 3 * row
            player = self.state[row_start]
            if player == self.CELL_EMPTY:
                continue
            # Assume they win, try to disprove
            is_winner = True
            for col in range(1, 3):
                if self.state[row_start + col] != player:
                    is_winner = False
                    break
            if is_winner:
                return player

        # Vertical
        for col in range(3):
            player = self.state[col]
            if player == self.CELL_EMPTY:
                continue
            # Assume they win, try to disprove
            is_winner = True
            for row in range(1, 3):
                if self.state[col + 3 * row] != player:
                    is_winner = False
                    break
            if is_winner:
                return player

        # Diagonal TL->BR
        player = self.state[0]
        if player is not self.CELL_EMPTY:
            # Assume they win, try to disprove
            is_winner = True
            for index in [4, 8]:
                if self.state[index] != player:
                    is_winner = False
                    break
            if is_winner:
                return player

        # Diagonal TR->BL
        player = self.state[2]
        if player is not self.CELL_EMPTY:
            # Assume they win, try to disprove
            is_winner = True
            for index in [4, 6]:
                if self.state[index] != player:
                    is_winner = False
                    break
            if is_winner:
                return player

        # Nobody wins
        return 0

    def is_over(self):
        """
        Returns true if the game is over
        """
        if len(self.get_valid_moves()) == 0:
            return True

        return self.get_winner() != 0

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
    