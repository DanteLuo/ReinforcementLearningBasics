import numpy as np
import matplotlib.pyplot as plt
import gym


# Q1
def bandit_one():
    return 8


def bandit_two():
    randm_num = np.random.rand()
    return 0 if randm_num > 0.12 else 100


def bandit_three():
    return np.random.uniform(-10.0, 35.0, 1)


def bandit_four():
    randm_num = np.random.uniform(0.0, 3.0, 1)
    if randm_num < 1:
        return 0
    elif randm_num < 2:
        return 20
    else:
        return np.random.randint(7, 19)


def learn_bandit_rewards(bandits, epsilon, num_episodes):
    num_arms = len(bandits)
    Q = np.zeros(num_arms)
    N = np.zeros(num_arms)
    Q_max = np.zeros(num_episodes)

    for i_episode in range(num_episodes):

        randm_num = np.random.rand()

        # epsilon-soft
        if randm_num > epsilon:
            bandit_index = np.argmax(Q)
        else:
            bandit_index = np.random.randint(4)

        reward = bandits[bandit_index]()
        N[bandit_index] += 1
        Q[bandit_index] += (reward - Q[bandit_index]) / N[bandit_index]
        Q_max[i_episode] = np.max(Q)

    plt.plot(range(1, 101), Q_max)
    plt.ylabel('Q_max')
    plt.xlabel('episodes')
    plt.show()

    return


# Q2:
def policy_evaluation(policy, env, gamma=0.95, theta=0.0001):

    env.reset()
    V = np.zeros(env.nS)
    A_state = np.zeros(env.nS,dtype=int)
    delta = theta+1.

    while delta>=theta:
        delta = 0.

        for state_id in env.P:

            value     = V[state_id]
            new_value = 0.
            max_v     = 0.

            for action_id in env.P[state_id]:

                action_value = 0.

                for transition_id in range(len(env.P[state_id][action_id])):

                    transition_prob = env.P[state_id][action_id][transition_id][0]
                    next_state_id   = env.P[state_id][action_id][transition_id][1]
                    reward          = env.P[state_id][action_id][transition_id][2]
                    # done            = env.P[state_id][action_id][transition_id][3]

                    # if done:
                    #     action_value += transition_prob*reward
                    # else:
                    action_value += transition_prob*(reward+gamma*V[next_state_id])

                if action_value > max_v:
                    max_v = action_value
                    A_state[state_id] = action_id

                new_value += action_value*policy[state_id][action_id]

            V[state_id] = new_value
            delta = max(delta,abs(V[state_id]-value))

    print(V)

    return V, A_state


def main():
    # Q1:
    bandits = [bandit_one, bandit_two, bandit_three, bandit_four]
    learn_bandit_rewards(bandits, 0.3, 100)


    # Q2:
    env = gym.make('FrozenLake-v0').unwrapped
    policy = np.ones([env.nS,env.nA])/env.nA
    policy_evaluation(policy,env)

    return


if __name__ == "__main__":
    main()