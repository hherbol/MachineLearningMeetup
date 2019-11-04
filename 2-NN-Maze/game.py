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
import numpy as np
from utils_imgs import *


class Game():
    HELD_MAZE, nBlocks = load_maze("maze.png")
    MAX_ITER = 100000
    # MAX_ITER = 1024

    def __init__(self):
        self.maze = copy.deepcopy(self.HELD_MAZE)
        self.px, self.py = 0, 0
        self.done = False
        self.final_x, self.final_y = len(self.maze[0]) - 1, len(self.maze) - 1
        self.MOVES = [
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, 0),
        ]
        self.ALL_MOVES = [(0, 0)] + self.MOVES
        self.colors, self.inv_colors = get_colors()
        self.iter = self.MAX_ITER

    def __repr__(self):
        state = self.get_state(rand_rotate=False)[0]
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
        return self.iter < 0

    def get_state(self, rand_rotate=False):
        # ONE IDEA: I could randomly rotate the maze, as we need to be able to
        # handle any orientation the same; however, we'll be biased to what we
        # see in the beginning.
        if rand_rotate:
            pos = [
                self.maze[self.py + self.dy][self.px + self.dx]
                if pos_chk(self.px + self.dx, self.py + self.dy, self.nBlocks)
                else B_WALL
                for self.dx, self.dy in self.ALL_MOVES
            ]
            rand = np.random.randint(4)
            return torch.Tensor(
                [[pos[0]] + [pos[1 + (i + rand) % 4] for i in range(4)]])
        else:
            return torch.Tensor([[
                self.maze[self.py + self.dy][self.px + self.dx]
                if pos_chk(self.px + self.dx, self.py + self.dy, self.nBlocks)
                else B_WALL
                for self.dx, self.dy in self.ALL_MOVES
            ]])

    def step(self, action, slow=False):
        reward = -0.4
        self.iter -= 1

        if action < 2:
            new_color = [
                B_VALID, B_BACKTRACK
            ][action]
            if self.maze[self.py][self.px] == new_color:
                reward = -0.8
            else:
                self.maze[self.py][self.px] = new_color
                # If we change the color to white (trying to game the system
                # by taking advantage of the white space bonus) we have a
                # penalty
                # if new_color == B_PATH:
                #     reward = -0.45
        else:
            self.dx, self.dy = self.MOVES[action - 2]
            # Check if valid move.  If not, do nothing
            in_bounds = pos_chk(self.px + self.dx, self.py + self.dy,
                                self.nBlocks)
            if in_bounds and\
                    self.maze[self.py + self.dy][self.px + self.dx] != B_WALL:
                self.px, self.py = self.px + self.dx, self.py + self.dy
                # If we move, we have a smaller penalty than normal
                reward = -0.2
                if self.maze[self.py][self.px] == B_PATH:
                    # If we moved to a white space, we have a small benefit
                    reward = 0.2
            elif not in_bounds:
                reward = -0.8
            else:
                reward = -0.8

        if self.is_finished():
            reward = 1.0

        if slow:
            save_maze(self.maze, name="steps")

        return reward, self.is_finished()

    def save_maze(self, name):
        save_maze(self.maze, name=name)

    # def SET_WHITE(self):
    #     self.step(0)

    def SET_GREEN(self):
        self.step(0)

    def SET_RED(self):
        self.step(1)

    # def SET_BLUE(self):
    #     self.step(3)

    def MOVE_UP(self):
        self.step(2)

    def MOVE_DOWN(self):
        self.step(3)

    def MOVE_LEFT(self):
        self.step(4)

    def MOVE_RIGHT(self):
        self.step(5)

    def play(self, model, slow=False, max_iter=1000):
        for i in range(max_iter):
            action = np.nanargmax(
                model(self.get_state()).detach().numpy()
            )
            self.step(action)
            if slow:
                save_maze(self.maze, name="solution")
                time.sleep(0.1)
            if self.is_finished():
                break


if __name__ == "__main__":
    g = Game()
    print(g)
    g.SET_GREEN()
    g.MOVE_RIGHT()
    print(g)
    g.SET_GREEN()
    g.MOVE_RIGHT()
    print(g)
    g.SET_GREEN()
    g.MOVE_DOWN()
    print(g)
