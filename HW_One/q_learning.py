import gym
import numpy as np


def q_learning(env, alpha=0.5, gamma=0.95, epsilon=0.1, num_episodes=500):

    # initialization
    Q = np.zeros([env.nS,env.nA])
    np.random.seed(42)

    for i_episode in range(num_episodes):
        # epsilon-soft
        randm_num = np.random.rand()
        Q_buf = Q[state_id][:]

        if randm_num > epsilon:
            action = np.argmax(Q_buf)
        else:
            action = np.random.randint(env.nA)


def main():
    env = gym.make('FrozenLake-v0').unwrapped
    q_learning(env)


if __name__ == "__main__":
    main()