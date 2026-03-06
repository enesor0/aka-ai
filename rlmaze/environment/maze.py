import logging
from enum import Enum, IntEnum

import numpy as np
import matplotlib.pyplot as plt


class Cell(IntEnum):
    EMPTY = 0
    OCCUPIED = 1
    CURRENT = 2


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Render(Enum):
    NONE = 0
    TRAINING = 1
    MOVES = 2


class Status(Enum):
    WIN = 0
    LOSE = 1
    PLAYING = 2


class Maze:
    """
    A maze with walls. An agent is placed at the start cell and must find the exit cell by moving through the maze.
    """

    actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    reward_exit = 10.0
    penalty_move = -0.05
    penalty_visited = -0.25
    penalty_impossible = -0.75

    def __init__(self, maze, start_cell, exit_cell=None, penalty_threshold=-5.0, render=Render.NONE):
        """Create a maze environment."""
        self.maze = np.array(maze)
        self.penalty_threshold = penalty_threshold
        self.minimum_reward = -0.5 * self.maze.size

        nrows, ncols = self.maze.shape
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]

        # Empty cells are where maze[row, col] == Cell.EMPTY
        self.empty_cells = [cell for cell in self.cells if self.maze[cell[::-1]] == Cell.EMPTY]

        self.__exit_cell = (ncols - 1, nrows - 1) if exit_cell is None else exit_cell

        if self.__exit_cell not in self.empty_cells:
            raise ValueError(f"Exit cell {self.__exit_cell} is not an empty cell")
        if self.maze[self.__exit_cell[::-1]] == Cell.OCCUPIED:
            raise ValueError(f"Exit cell {self.__exit_cell} is occupied")

        # Optional: keep exit out of candidate empty cells for random starts
        if self.__exit_cell in self.empty_cells:
            self.empty_cells.remove(self.__exit_cell)

        self.__render = render
        self.__ax1 = None
        self.__ax2 = None

        self.reset(start_cell)

    def reset(self, start_cell=(0, 0)):
        """Reset the maze to its initial state and place the agent at start_cell."""
        if start_cell not in self.cells:
            raise Exception(f"Error: start cell at {start_cell} is not inside maze")
        if self.maze[start_cell[::-1]] == Cell.OCCUPIED:
            raise Exception(f"Error: start cell at {start_cell} is not free")
        if start_cell == self.__exit_cell:
            raise Exception(f"Error: start- and exit cell cannot be the same {start_cell}")

        self.__previous_cell = self.__current_cell = start_cell
        self.__total_reward = 0.0
        self.__visited = set()

        if self.__render in (Render.TRAINING, Render.MOVES) and self.__ax1 is not None:
            nrows, ncols = self.maze.shape
            self.__ax1.clear()
            self.__ax1.set_xticks(np.arange(0.5, ncols, step=1))
            self.__ax1.set_xticklabels([])
            self.__ax1.set_yticks(np.arange(0.5, nrows, step=1))
            self.__ax1.set_yticklabels([])
            self.__ax1.grid(True)
            self.__ax1.plot(*self.__current_cell, "rs", markersize=30)
            self.__ax1.text(*self.__current_cell, "Start", ha="center", va="center", color="white")
            self.__ax1.plot(*self.__exit_cell, "gs", markersize=30)
            self.__ax1.text(*self.__exit_cell, "Exit", ha="center", va="center", color="white")
            self.__ax1.imshow(self.maze, cmap="binary")
            self.__ax1.get_figure().canvas.draw()
            self.__ax1.get_figure().canvas.flush_events()
        return self.__current_cell

