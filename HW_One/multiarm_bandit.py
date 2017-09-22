import numpy as np
import matplotlib.pyplot as plt


def main():
    bandits = [bandit_one, bandit_two, bandit_three, bandit_four]
    for epsilon in [0.3,0.1,0.03]:
        learn_bandit_rewards(bandits, epsilon, 100)
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
        return np.random.randint(7,19)


def learn_bandit_rewards(bandits, epsilon, num_episodes):
    np.random.seed(42)  # seed for random number
    num_arms = len(bandits)
    Q = np.zeros(num_arms)
    N = np.zeros(num_arms)
    Q_max = np.zeros(num_episodes)

    for i_episode in range(num_episodes):

        randm_num = np.random.rand()

        # epsilon-soft
        if randm_num>epsilon:
            bandit_index = np.argmax(Q)
        else:
            bandit_index = np.random.randint(len(bandits))

        reward = bandits[bandit_index]()
        N[bandit_index] += 1
        Q[bandit_index] += (reward-Q[bandit_index])/N[bandit_index]
        Q_max[i_episode] = np.max(Q)

    plt.plot(range(1,num_episodes+1),Q_max)
    plt.ylabel('Q_max')
    plt.xlabel('episodes')
    plt.show()

if __name__ == "__main__":
    main()

