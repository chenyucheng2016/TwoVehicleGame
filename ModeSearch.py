import math
import random

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        actions = self.state.get_legal_actions()
        for action in actions:
            if not any(child.state == self.state.move(action) for child in self.children):
                new_state = self.state.move(action)
                child_node = Node(new_state, parent=self)
                self.children.append(child_node)
                return child_node

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def best_action(self):
        return self.best_child(c_param=0)

class MCTS:
    def __init__(self, n_simulations=1000):
        self.n_simulations = n_simulations

    def search(self, initial_state):
        root = Node(state=initial_state)

        for _ in range(self.n_simulations):
            node = self._select(root)
            reward = self._simulate(node.state)
            node.backpropagate(reward)

        return root.best_action().state

    def _select(self, node):
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child()
        return node

    def _simulate(self, state):
        current_state = state
        while not current_state.is_terminal():
            action = random.choice(current_state.get_legal_actions())
            current_state = current_state.move(action)
        return current_state.get_reward()

# Example usage with a simple game-like structure
class GameState:
    def get_legal_actions(self):
        # Define legal actions for the state
        pass
    
    def move(self, action):
        # Return the new state after applying the action
        pass
    
    def is_terminal(self):
        # Return True if the state is terminal (end of game)
        pass
    
    def get_reward(self):
        # Return the reward for the current state (win, lose, draw)
        pass

# Assuming a GameState object is provided
initial_state = GameState()
mcts = MCTS(n_simulations=1000)
best_next_state = mcts.search(initial_state)
