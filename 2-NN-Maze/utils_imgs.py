from PIL import Image


B_WALL = 0
B_PATH = 1
B_VALID = 2
B_BACKTRACK = 3
B_ENDPOINT = 4
B_SOLUTION = 5


def get_colors():
    '''
    Colors map that the maze will use:
        0 - Black - A wall
        1 - White - A space to travel in the maze
        2 - Green - A valid solution of the maze
        3 - Red - A backtracked position during maze solving
        4 - Blue - Start and Endpoints of the maze

    **Returns**

        color_map: *dict, int, tuple*
            A dictionary that will correlate the integer key to
            a color.
    '''
    COLORS = {
        B_WALL: (0, 0, 0),
        B_PATH: (255, 255, 255),
        B_VALID: (0, 255, 0),
        B_BACKTRACK: (255, 0, 0),
        B_ENDPOINT: (0, 0, 255),
        B_SOLUTION: (255, 255, 0),
    }
    INV_COLORS = {v: k for k, v in COLORS.items()}
    return COLORS, INV_COLORS


def save_maze(maze, blockSize=10, name="maze"):
    '''
    This will save a maze object to a file.

    **Parameters**

        maze: *list, list, int*
            A list of lists, holding integers specifying the different aspects
            of the maze:
                0 - Black - A wall
                1 - White - A space to travel in the maze
                2 - Green - A valid solution of the maze
                3 - Red - A backtracked position during maze solving
                4 - Blue - Start and Endpoints of the maze
        blockSize: *int, optional*
            How many pixels each block is comprised of.
        name: *str, optional*
            The name of the maze.png file to save.

    **Returns**

        None
    '''
    nBlocks = len(maze)
    dims = nBlocks * blockSize
    colors, inv_colors = get_colors()

    # Verify that all values in the maze are valid colors.
    ERR_MSG = "Error, invalid maze value found!"
    assert all([x in colors.keys() for row in maze for x in row]), ERR_MSG

    img = Image.new("RGB", (dims, dims), color=0)

    # Parse "maze" into pixels
    for jx in range(nBlocks):
        for jy in range(nBlocks):
            x = jx * blockSize
            y = jy * blockSize
            for i in range(blockSize):
                for j in range(blockSize):
                    img.putpixel((x + i, y + j), colors[maze[jx][jy]])

    if not name.endswith(".png"):
        name += ".png"
    img.save("%s" % name)


def load_maze(filename, blockSize=10):
    '''
    This will read a maze from a png file into a 2d list with values
    corresponding to the known color dictionary.

    **Parameters**

        filename: *str*
            The name of the maze.png file to load.
        blockSize: *int, optional*
            How many pixels each block is comprised of.

    **Returns**

        maze: *list, list, int*
            A 2D array holding integers specifying each block's color.
    '''
    if ".png" in filename:
        filename = filename.split(".png")[0]
    img = Image.open(filename + ".png")
    dims, _ = img.size
    nBlocks = int(dims / blockSize)
    colors, inv_colors = get_colors()
    color_map = {v: k for k, v in colors.items()}

    maze = [[0 for x in range(nBlocks)] for y in range(nBlocks)]

    for i, x in enumerate(range(0, dims, dims // nBlocks)):
        for j, y in enumerate(range(0, dims, dims // nBlocks)):
            px = x
            py = y
            maze[i][j] = color_map[img.getpixel((px, py))]

    return maze, nBlocks


def pos_chk(x, y, nBlocks):
    '''
    Validate if the coordinates specified (x and y) are within the maze.

    **Parameters**

        x: *int*
            An x coordinate to check if it resides within the maze.
        y: *int*
            A y coordinate to check if it resides within the maze.
        nBlocks: *int*
            How many blocks wide the maze is.  Should be equivalent to
            the length of the maze (ie. len(maze)).

    **Returns**

        valid: *bool*
            Whether the coordiantes are valid (True) or not (False).
    '''
    return x >= 0 and x < nBlocks and y >= 0 and y < nBlocks


if __name__ == "__main__":
    pass
