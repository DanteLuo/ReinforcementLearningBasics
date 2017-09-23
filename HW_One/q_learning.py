import gym
import numpy as np


def q_learning(env, alpha=0.5, gamma=0.95, epsilon=0.1, num_episodes=500):

    # initialization
    Q = np.zeros([env.nS,env.nA])
    np.random.seed(42)

    for i_episode in range(num_episodes):

        env.reset()
        state_id = 0 # how to initialize in general case, where no starting state?
        num_step = 0
        done = False

        while not done:

            # epsilon-soft
            randm_num = np.random.rand()

            if randm_num > epsilon:
                action = np.argmax(Q[state_id][:])
            else:
                action = np.random.randint(env.nA)

            next_state_id, reward, done, info = env.step(action)

            # update Q
            Q[state_id][action] += alpha*(reward+gamma*np.max(Q[next_state_id][:])-Q[state_id][action])

            state_id = next_state_id
            num_step += 1

        print("Episode finished after {} timesteps".format(num_step))

    print(Q)

    return Q


def main():
    env = gym.make('FrozenLake-v0').unwrapped
    q_learning(env)


if __name__ == "__main__":
    main()