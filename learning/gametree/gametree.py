"""
Functions for generating and searching game treess
"""
import math
import random
import pickle
from learning.environments.tic_tac_toe import TicTacToeGame

def main():
    # generate_tic_tac_toe_game_tree()
    test_negamax_search()

def compute_game_tree(game, root_state):
    """
    Compute the game tree rooted at root_state
    """
    root = GameTreeNode(root_state)
    game.set_state(root_state)

    if game.is_over():
        return root

    actions = game.get_valid_moves()

    for action in actions:
        game.set_state(root_state)
        game.move(action)
        successor_state = game.get_state()
        root.add_child(action, compute_game_tree(game, successor_state))

    game.set_state(root_state)
    return root

def generate_tic_tac_toe_game_tree():
    """
    Generate a full game tree for tic tac toe
    """
    out_file = open("data/game_tree.txt", "w")
    game = TicTacToeGame()
    # Handy test state for debugging
    game.set_state([0, 2, 0,
                    0, 1, 2,
                    0, 1, 0])
    root = compute_game_tree(game, game.get_state())
    tree_string = root.to_string(game, 4, 0)
    out_file.write("Tree size: %d\n" % root.get_tree_size())
    out_file.write("Max tree depth: %d\n" % root.get_max_tree_depth())
    out_file.write("Leaf nodes: %d\n" % root.get_leaf_count())
    out_file.write("Full tree:\n")
    out_file.write(tree_string)

def get_negamax_action(game):
    """
    Return the infinite depth negamax action!
    """
    state = game.get_state()
    root = compute_game_tree(game, state)
    player = game.get_active_player()
    action, value = negamax_search(game, root, player)
    print("Value %d:" %value)
    game.set_state(state)
    return action

def negamax_search(game, node, player):
    """
    Cribbed mercilessly from https://en.wikipedia.org/wiki/negamax
    """
    game.set_state(node.state)

    if game.is_over():
        winner = game.get_winner()
        if winner == 0:
            # It's a draw!
            return -1, 0
        if winner == player:
            return -1, 1
        else:
            return -1, -1

    adversary = 2 if player == 1 else 1
    best_value = -1 * math.inf
    best_actions = []
    for action, child in node.children.items():
        _, value = negamax_search(game, child, adversary)
        value = -1 * value
        if value > best_value:
            best_value = value
            best_actions = [action]
        elif value == best_value:
            best_actions.append(action)
    return random.choice(best_actions), best_value


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

class GameTreeNode:
    """
    Node in a game tree
    """

    def __init__(self, state):
        self.children = {}
        self.state = state

    def add_child(self, action, child_node):
        """
        Add child_node along the action edge
        """
        self.children[action] = child_node

    def to_string(self, game, indent, level):
        """Pretty print the tree rooted at this node.

        Args:
            game:   game used to interpret state
            indent: number of spaces per indentation level
            level:  indentation level
        """
        game.set_state(self.state)
        total_indent = indent * level
        indent_string = " " * total_indent
        result = indent_string + "Node:\n"
        result = result + indent_string + "State:\n"
        result = result + game.to_string(total_indent) + "\n"
        result = result + indent_string + "Children:\n"
        if len(self.children) == 0:
            result = result + indent_string + "None\n"
        else:
            for action, child in self.children.items():
                result = result + indent_string + "Action %d ->\n" % action
                result = result + child.to_string(game, indent, level + 1)
        return result

    def get_tree_size(self):
        """
        Return the size of the tree rooted at this node
        """
        result = 1
        for _, child in self.children.items():
            result = result + child.get_tree_size()
        return result

    def get_max_tree_depth(self):
        """
        Return the max depth of the tree rooted at this node
        """
        max_depth = 0
        for _, child in self.children.items():
            depth = child.get_max_tree_depth()
            if depth > max_depth:
                max_depth = depth
        return max_depth + 1

    def get_leaf_count(self):
        """
        Return the number of leaves int the tree rooted at this node
        """
        if len(self.children) == 0:
            return 1

        leaves = 0
        for _, child in self.children.items():
            leaves = leaves + child.get_leaf_count()
        return leaves

if __name__ == "__main__":
    main()