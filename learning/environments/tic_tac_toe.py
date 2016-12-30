"""
TicTacToe framework
"""

def main():
    """Handy entry point for experimentation"""
    game = TicTacToeGame()
    game.set_state([0, 0, 0,
                    0, 1, 2,
                    0, 0, 0])
    print(game)

class TicTacToeGame:
    """Simple representation of a TicTacToe game"""

    CELL_EMPTY = 0
    CELL_P1 = 1
    CELL_P2 = 2

    def __init__(self):
        """Default contructor"""
        self.state = []
        self.turn = 0
        self.reset()

    def get_active_player(self):
        """Return the player whose turn it is"""
        return self.get_active_player_for_state(self.state)

    def get_active_player_for_state(self, state):
        """Returns the active player

        Args:
            state: the current state
        """
        p1_moves = 0
        p2_moves = 0
        for cell in state:
            if cell == self.CELL_P1:
                p1_moves = p1_moves + 1
            elif cell == self.CELL_P2:
                p2_moves = p2_moves + 1
        return 2 if p1_moves > p2_moves else 1


    def get_state(self):
        """Returns a copy of the raw state of the board. Turn number is not a part of the state."""
        return list(self.state)

    def get_state_successor(self, state, action):
        """Returns the state resulting from taking an action

        Args:
            state: The current state
            action: The action to take
        Returns:
            The successor state
        """
        # Work out whose move it is
        p1_moves = 0
        p2_moves = 0
        for cell in state:
            if cell == self.CELL_P1:
                p1_moves = p1_moves + 1
            elif cell == self.CELL_P2:
                p2_moves = p2_moves + 1
        player = 2 if p1_moves > p2_moves else 1
        successor = list(state)
        successor[action] = player
        return successor

    def get_state_is_over(self, state):
        """Returns true if the game is over

        Args:
            state: The current state
        """
        if len(self.get_state_valid_moves(state)) == 0:
            return True

        return self.get_state_winner(state) != 0

    def get_state_valid_moves(self, state):
        """Gets the valid moves or actions
        
        Args:
            state: The current state
        Returns:
            A list of the valid moves represented as ints
        """
        return [move for move in range(9) if state[move] == self.CELL_EMPTY]

    def get_state_winner(self, state):
        """Gets the winner in the provided state
        Args:
            state: The current state
        Returns:
            If there is a winner, returns the player number, otherwise returns 0
        """
        # Horizontal
        for row in range(3):
            row_start = 3 * row
            player = state[row_start]
            if player == self.CELL_EMPTY:
                continue
            # Assume they win, try to disprove
            is_winner = True
            for col in range(1, 3):
                if state[row_start + col] != player:
                    is_winner = False
                    break
            if is_winner:
                return player

        # Vertical
        for col in range(3):
            player = state[col]
            if player == self.CELL_EMPTY:
                continue
            # Assume they win, try to disprove
            is_winner = True
            for row in range(1, 3):
                if state[col + 3 * row] != player:
                    is_winner = False
                    break
            if is_winner:
                return player

        # Diagonal TL->BR
        player = state[0]
        if player is not self.CELL_EMPTY:
            # Assume they win, try to disprove
            is_winner = True
            for index in [4, 8]:
                if state[index] != player:
                    is_winner = False
                    break
            if is_winner:
                return player

        # Diagonal TR->BL
        player = state[2]
        if player is not self.CELL_EMPTY:
            # Assume they win, try to disprove
            is_winner = True
            for index in [4, 6]:
                if state[index] != player:
                    is_winner = False
                    break
            if is_winner:
                return player

        # Nobody wins
        return 0

    def get_valid_moves(self):
        """Gets the valid moves for the game's internal state.

        Returns:
            A list of the valid moves represented as ints.
        """ 
        return self.get_state_valid_moves(self.state)

    def reset(self):
        """Resets the internal state of the game"""
        self.set_state([self.CELL_EMPTY] * 9)

    def set_state(self, state):
        """Sets the internal state of the board.

        Args:
            state: The new state of the board.
        """
        self.state = list(state)
        # The turn number is the number non-empty cells
        self.turn = 9 - self.state.count(self.CELL_EMPTY)

    def get_winner(self):
        """Gets the winner for the current state of the game.
        
        Returns:
            The player number of the winner according to the 
            game's internal state, or 0 if there is no winner.
        """
        return self.get_state_winner(self.state)

    def is_over(self):
        """Queries whether the game is over.

        Returns:
            True if the internal state of the game has no valid
            moves or has a winner.
        """
        return self.get_state_is_over(self.state)

    def move(self, action):
        """Make a move in the game's internal state.

        Args:
            action: The action or move to take.
        """
        self.state = self.get_state_successor(self.state, action)
        self.turn = self.turn + 1

    def state_to_string(self, state, indent):
        """Stringify a state with indent spaces on each line.

        Args:
            state: The state to stringify
            indent: The number of spaces to include at the start of each line
        Returns:
            The stringified state.
        """
        indentation = " " * indent
        result = indentation
        for (index, cell) in enumerate(state):
            if cell == self.CELL_EMPTY:
                result += str(index)
            elif cell == self.CELL_P1:
                result += "X"
            elif cell == self.CELL_P2:
                result += "O"
            # Include a line break at the end of each row
            if index % 3 == 2:
                result += "\n" + indentation
        return result

    def __repr__(self):
        """Stringifies the internal game state

        Returns:
            The stringified internal state
        """
        result = self.state_to_string(self.state, 0)
        # Convert to human friendly turn number
        result += "Turn: %d" % (self.turn + 1)
        return result

if __name__ == "__main__":
    main()
