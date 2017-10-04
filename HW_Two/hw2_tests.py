import numpy as np
from hw2 import dyna_q_learning


class GridWorld:
    def __init__(self):
        self.nS = 25
        self.nA = 4
        self.row = 5
        self.col = 5
        self.model = [[ 1,  0,  0,  0, 0],
                      [ 0, -1,  0, -1, 0],
                      [-1,  0,  0, -1, 0],
                      [-1,  0, -1, -1, 0],
                      [ 0,  0,  0,  0, 2]]
        self.currentState = [0, 0]

    def seed(self, seed_num):
        return

    def getvalue(self):
        return self.model[self.currentState[0]][self.currentState[1]]

    def notdone(self):
        return True if self.getvalue() != -1 and self.getvalue() != 2 else False

    def step(self, action):
        if action == 0:
            if self.currentState[0] > 0 and self.notdone():
                self.currentState[0] -= 1
        elif action == 1:
            if self.currentState[1] > 0 and self.notdone():
                self.currentState[1] -= 1
        elif action == 2:
            if self.currentState[0] < self.row-1 and self.notdone():
                self.currentState[0] += 1
        elif action == 3:
            if self.currentState[1] < self.col-1 and self.notdone():
                self.currentState[1] += 1

        done = False
        reward = -1
        info = 'Stable step.'
        if self.getvalue() == -1:
            reward -= 3
            done = True
            info = 'Dang it!'
        elif self.getvalue() == 2:
            reward += 100
            done = True
            info = 'Goal!'

        observation = self.currentState[0]*self.row+self.currentState[1]

        return observation, reward, done, info

    def reset(self):
        self.currentState = [0, 0]


def main():
    env = GridWorld()
    policy = dyna_q_learning(env, 150)

if __name__ == '__main__':
    main()