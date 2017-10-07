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
                equal_num = 0
                action_candidates = np.zeros(env.nA, dtype=int)
                action = np.argmax(Q[state_id][:])
                action_candidates[equal_num] = action
                equal_num += 1

                for action_choice_id in range(env.nA):
                    if Q[state_id][action_choice_id] == Q[state_id][action] and action_choice_id != action:
                        action_candidates[equal_num] = action_choice_id
                        equal_num += 1

                if equal_num > 1:
                    rand_action = np.random.randint(0, equal_num - 1)
                    action = action_candidates[rand_action]
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