import collections
import math

import numpy as np


class UCTNode:
    def __init__(self, env, action, parent=None):
        self.env = env
        self.state = state
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.child_priors = np.zeros([362], dtype=np.float32)
        self.child_total_value = np.zeros([362], dtype=np.float32)
        self.child_number_visits = np.zeros([362], dtype=np.float32)

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * (
            self.child_priors / (1 + self.child_number_visits)
        )

    def best_child(self):
        return np.argmax(self.child_Q() + self.child_U())

    def select_leaf(self):
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def maybe_add_child(self, move):
        if move not in self.children:
            self.children[move] = UCTNode(self.game_state.play(move), move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value_estimate * self.game_state.to_play
            current = current.parent


class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def UCT_search(game_state, num_reads):
    root = UCTNode(game_state, move=None, parent=DummyNode())
    for _ in range(num_reads):
        leaf = root.select_leaf()
        child_priors, value_estimate = NeuralNet.evaluate(leaf.game_state)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
    return np.argmax(root.child_number_visits)


class NeuralNet:
    @classmethod
    def evaluate(self, game_state):
        return np.random.random([362]), np.random.random()


class GameState:
    def __init__(self, to_play=1):
        self.to_play = to_play

    def play(self, move):
        return GameState(-self.to_play)


num_reads = 10000
import time

tick = time.time()
UCT_search(GameState(), num_reads)
tock = time.time()
print("Took %s sec to run %s times" % (tock - tick, num_reads))
import resource

print("Consumed %sB memory" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
