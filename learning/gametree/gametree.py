"""
Utilities for generating and searching game treess
"""
import math
import random
import pickle
from learning.environments.tic_tac_toe import TicTacToeGame

def main():
    """
    Handy entry point for testing stuff
    """
    pass
    # generate_tic_tac_toe_game_tree()

def generate_game_tree(game, root_state):
    """
    Compute the game tree rooted at root_state
    """
    root = GameTreeNode(root_state)

    if game.get_state_is_over(root_state):
        return root

    actions = game.get_state_valid_moves(root_state)

    for action in actions:
        successor_state = game.get_state_successor(root_state, action)
        root.add_child(action, generate_game_tree(game, successor_state))

    return root

def generate_tic_tac_toe_game_tree():
    """
    Generate a full game tree for tic tac toe
    """
    out_file = open("data/tic_tac_toe_game_tree.txt", "w")
    game = TicTacToeGame()
    # # Handy test state for debugging
    # game.set_state([1, 2, 1,
    #                 0, 2, 2,
    #                 0, 1, 1])
    root = generate_game_tree(game, game.get_state())
    tree_string = root.to_string(game, 4, 0)
    out_file.write("Tree size: %d\n" % root.get_tree_size())
    out_file.write("Max tree depth: %d\n" % root.get_max_tree_depth())
    out_file.write("Leaf nodes: %d\n" % root.get_leaf_count())
    out_file.write("Full tree:\n")
    out_file.write(tree_string)
    out_file.close()

    pickle_file = open("data/tic_tac_toe_game_tree.pkl", "wb")
    pickle.dump(root, pickle_file)
    pickle_file.close()

def get_negamax_action(game):
    """
    Return the infinite depth negamax action!
    """
    state = game.get_state()
    root = generate_game_tree(game, state)
    player = game.get_active_player()
    action, value = negamax_search(game, root, player)
    print("Value %d:" %value)
    return action

def negamax_search(game, node, player):
    """
    Cribbed mercilessly from https://en.wikipedia.org/wiki/negamax
    """
    state = node.state
    if game.get_state_is_over(state):
        winner = game.get_state_winner(state)
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

    def get_leaf_count(self):
        """
        Returns the number of leaves in the tree rooted at this node
        """
        if len(self.children) == 0:
            return 1

        leaves = 0
        for _, child in self.children.items():
            leaves = leaves + child.get_leaf_count()
        return leaves

    def get_max_tree_depth(self):
        """
        Returns the max depth of the tree rooted at this node
        """
        max_depth = 0
        for _, child in self.children.items():
            depth = child.get_max_tree_depth()
            if depth > max_depth:
                max_depth = depth
        return max_depth + 1

    def get_tree_size(self):
        """
        Return the total number of nodes in the tree rooted at this node
        """
        result = 1
        for _, child in self.children.items():
            result = result + child.get_tree_size()
        return result

    def to_string(self, game, indent, level):
        """Produce a nicely formatted representation of this node.

        Args:
            game: game used to interpret state
            indent: number of spaces per indentation level
            level: indentation level
        """
        total_indent = indent * level
        indent_string = " " * total_indent
        result = indent_string + "Node:\n"
        result = result + indent_string + "State:\n"
        result = result + game.state_to_string(self.state, total_indent) + "\n"
        result = result + indent_string + "Children:\n"
        if len(self.children) == 0:
            result = result + indent_string + "None\n"
        else:
            for action, child in self.children.items():
                result = result + indent_string + "Action %d ->\n" % action
                result = result + child.to_string(game, indent, level + 1)
        return result

if __name__ == "__main__":
    main()
