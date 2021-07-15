import gym
from gym import error, spaces, utils
from gym.utils import seeding

import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

REWARDS = {
    b'S': 0,
    b'W' : 0,
    b'R' : -1,
    b'L' : -5,
    b'G' : 10,
}

MAP = {
    "4x4:1": [
        "SWWR",
        "WWRL",
        "RWRL",
        "LRWG"
    ],
    "4x4:2": [
        "SWRL",
        "RWWR",
        "LRWW",
        "LRWG"
    ],
    "4x4:3": [
        "SRLL",
        "WWRR",
        "RWWW",
        "LRWG"
    ],
    "4x4:4": [
        "SWWR",
        "WLWL",
        "WWRL",
        "LRWG"
    ],
    "4x4:5": [
        "SWWL",
        "LRWW",
        "RWRW",
        "WRLG"
    ]
}

class WaterworldEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4:1"):
        desc = MAP[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (-5, 10)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b'GL'
            reward = REWARDS[newletter]
            return newstate, reward, done


        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GL':
                        li.append((1.0, s, 0, True))
                    else:
                        li.append((
                            1., *update_probability_matrix(row, col, a)
                        ))

        super(WaterworldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
