import numpy as np
import matplotlib.pyplot as plt


def main():
    bandits = [bandit_one, bandit_two, bandit_three, bandit_four]
    num_episodes = 1000
    for epsilon in [0.3,0.1,0.03]:
        print(epsilon)
        Q, Q_max = learn_bandit_rewards(bandits, epsilon, num_episodes)
        plt.plot(range(1, num_episodes + 1), Q_max)

    plt.ylabel('Q_max')
    plt.xlabel('episodes')
    plt.legend(['epsilon = 0.3','epsilon = 0.1','epsilon = 0.03'])
    plt.show()

    return


## definition for bandits
def bandit_one():
    return 8


def bandit_two():
    randm_num = np.random.rand()
    return 0 if randm_num>0.12 else 100


def bandit_three():
    return np.random.uniform(-10.0, 35.0, 1)


def bandit_four():
    randm_num = np.random.uniform(0.0,3.0,1)
    if randm_num<1:
        return 0
    elif randm_num<2:
        return 20
    else:
        return np.random.randint(8,19)


def learn_bandit_rewards(bandits, epsilon, num_episodes):
    np.random.seed(42)  # seed for random number
    num_arms = len(bandits)
    Q = np.zeros(num_arms)
    N = np.zeros(num_arms)
    Q_max = np.zeros(num_episodes)

    for i_episode in range(num_episodes):
        print(i_episode)
        Q_max[i_episode] = np.max(Q)

        randm_num = np.random.rand()

        # epsilon-soft
        if randm_num > epsilon:
            equal_num = 0
            action_candidates = np.zeros(num_arms,dtype=int)
            bandit_index = np.argmax(Q)
            action_candidates[equal_num] = bandit_index
            equal_num += 1

            for action_choice_id in range(num_arms):
                if Q[action_choice_id] == Q[bandit_index] and action_choice_id != bandit_index:
                    action_candidates[equal_num] = action_choice_id
                    equal_num += 1

            if equal_num > 1:
                rand_action = np.random.randint(0,equal_num-1)
                bandit_index = action_candidates[rand_action]

            # bandit_index_set = np.argwhere(Q == np.max(Q))
            # index = np.random.randint(0,len(bandit_index_set))
            # bandit_index = int(bandit_index_set[index])
        else:
            bandit_index = np.random.randint(len(bandits))

        reward = bandits[bandit_index]()
        N[bandit_index] += 1
        Q[bandit_index] += (reward-Q[bandit_index])/N[bandit_index]

    return Q, Q_max

if __name__ == "__main__":
    main()

