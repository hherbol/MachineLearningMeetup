'''
Goal is to train a NN to "solve" a maze in the least steps possible.
The neural network will see the following:

    1. Current position color
    2. Neighbouring position colors

For each move, the NN will have the following options available:

    1. Set current color to WHITE/RED/GREEN/BLUE
    2. Move left, right, up, or down
'''
import torch
import numpy as np
from utils_imgs import *


def play_maze(model, maze, nBlocks, max_game_loop=5000):
    colors, inv_colors = get_colors()
    MOVES = [
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0)
    ]

    ALL_MOVES = [(0, 0)] + MOVES

    current_position = (0, 0)
    final_position = (len(maze) - 1, len(maze) - 1)

    count = 0
    while current_position != final_position and count < max_game_loop:
        count += 1
        px, py = current_position
        local_vision = [
            maze[py + dy][px + dx]
            if pos_chk(px + dx, py + dy, nBlocks)
            else B_WALL
            for dx, dy in ALL_MOVES
        ]
        choice = model(torch.Tensor(local_vision))
        # Choice list will be:
        #   0 - Set WHITE
        #   1 - Set GREEN
        #   2 - Set RED
        #   3 - Set BLUE
        #   4 - Move UP
        #   5 - Move DOWN
        #   6 - Move LEFT
        #   7 - Move RIGHT
        choice = np.nanargmax(choice.detach().numpy())
        if choice < 4:
            maze[py][px] = [
                B_PATH, B_VALID, B_BACKTRACK, B_ENDPOINT
            ][choice]
        else:
            dx, dy = MOVES[choice - 4]
            # Check if valid move.  If not, do nothing
            if pos_chk(px + dx, py + dy, nBlocks) and\
                    maze[py + dy][px + dx] != B_WALL:
                current_position = (px + dx, py + dy)
        count += 1

    # SET THE LAST POSITION TO BE YELLOW
    px, py = current_position
    maze[py][px] = B_SOLUTION

    return count, maze


if __name__ == "__main__":
    count, solution = play_maze()
