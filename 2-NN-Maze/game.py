'''
Goal is to train a NN to "solve" a maze in the least steps possible.
The neural network will see the following:

    1. Current position color
    2. Neighbouring position colors

For each move, the NN will have the following options available:

    1. Set current color to WHITE/RED/GREEN/BLUE
    2. Move left, right, up, or down
'''
import copy
import time
import torch
from utils_imgs import *


class Game():
    HELD_MAZE, nBlocks = load_maze("maze.png")
    MAX_ITER = 100000

    def __init__(self):
        self.maze = copy.deepcopy(self.HELD_MAZE)
        self.px, self.py = 0, 0
        self.done = False
        self.final_x, self.final_y = len(self.maze[0]) - 1, len(self.maze) - 1
        self.MOVES = [
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0)
        ]
        self.ALL_MOVES = [(0, 0)] + self.MOVES
        self.colors, self.inv_colors = get_colors()
        self.iter = self.MAX_ITER

    def __repr__(self):
        state = self.get_state()[0]
        mid, up, down, left, right =\
            [["#", "W", "G", "R", "B"][int(i)] for i in state]
        labels = (up, left, mid, right, down)
        return '''
# %s #
%s %s %s
# %s #
''' % labels

    def reset(self):
        self.maze = copy.deepcopy(self.HELD_MAZE)
        self.px, self.py = 0, 0
        self.done = False
        self.iter = self.MAX_ITER

    def is_finished(self):
        self.done = (self.px, self.py) == (self.final_x, self.final_y)
        return self.done

    def timed_out(self):
        return self.iter <= 0

    def get_state(self):
        return torch.Tensor([[
            self.maze[self.py + self.dy][self.px + self.dx]
            if pos_chk(self.px + self.dx, self.py + self.dy, self.nBlocks)
            else B_WALL
            for self.dx, self.dy in self.ALL_MOVES
        ]])

    def step(self, action, slow=False):
        reward = -0.4
        self.iter -= 1

        if action < 4:
            self.maze[self.py][self.px] = [
                B_PATH, B_VALID, B_BACKTRACK, B_ENDPOINT
            ][action]
            reward = -0.45
        else:
            self.dx, self.dy = self.MOVES[action - 4]
            # Check if valid move.  If not, do nothing
            in_bounds = pos_chk(self.px + self.dx, self.py + self.dy,
                                self.nBlocks)
            if in_bounds and\
                    self.maze[self.py + self.dy][self.px + self.dx] != B_WALL:
                self.px, self.py = self.px + self.dx, self.py + self.dy
            elif not in_bounds:
                reward = -0.8
            else:
                reward = -0.8

        if self.is_finished():
            reward = 1.0

        if slow:
            save_maze(self.maze, name="steps")

        return torch.Tensor([reward]), self.is_finished()

    def SET_WHITE(self):
        self.step(0)

    def SET_GREEN(self):
        self.step(1)

    def SET_RED(self):
        self.step(2)

    def SET_BLUE(self):
        self.step(3)

    def MOVE_UP(self):
        self.step(4)

    def MOVE_DOWN(self):
        self.step(5)

    def MOVE_LEFT(self):
        self.step(6)

    def MOVE_RIGHT(self):
        self.step(7)

    def play(self, model, slow=False, max_iter=1000):
        for i in range(max_iter):
            self.step(
                np.nanargmax(
                    model(self.get_state()).detach().numpy()
                ))
            if slow:
                save_maze(self.maze, name="solution")
                time.sleep(0.1)
            if self.is_finished():
                break


if __name__ == "__main__":
    g = Game()
    print(g.maze[0][:5])
    print(g)
    g.SET_GREEN()
    g.MOVE_DOWN()
    print(g)
    g.SET_GREEN()
    g.MOVE_DOWN()
    print(g)
