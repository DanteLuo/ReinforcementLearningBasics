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
        return np.random.randint(8, 19)


def learn_bandit_rewards(bandits, epsilon, num_episodes):
    np.random.seed(42)  # seed for random number
    num_arms = len(bandits)
    Q = np.zeros(num_arms)
    N = np.zeros(num_arms)
    Q_max = np.zeros(num_episodes)

    for i_episode in range(num_episodes):

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
        else:
            bandit_index = np.random.randint(len(bandits))

        reward = bandits[bandit_index]()
        N[bandit_index] += 1
        Q[bandit_index] += (reward - Q[bandit_index]) / N[bandit_index]

    print(Q)

    # return Q,Q_max
    return Q


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

    return V


# Q3:
def value_iteration(env, gamma=0.95, theta=0.0001):
    delta = theta + 1.
    V = np.zeros(env.nS)
    Q = np.zeros([env.nS,env.nA])
    A_state = np.zeros(env.nS,dtype=int)

    while delta>=theta:
        delta = 0.

        for state_id in range(env.nS):

            value     = V[state_id]
            new_value = 0.

            for action_id in range(env.nA):

                action_value = 0.

                for transition_id in range(len(env.P[state_id][action_id])):

                    transition_prob = env.P[state_id][action_id][transition_id][0]
                    next_state_id   = env.P[state_id][action_id][transition_id][1]
                    reward          = env.P[state_id][action_id][transition_id][2]

                    action_value += transition_prob*(reward+gamma*V[next_state_id])

                Q[state_id][action_id] = action_value

                if action_value > new_value:
                    new_value = action_value
                    A_state[state_id] = action_id # getting the argmax action

            V[state_id] = new_value
            delta = max(delta,abs(V[state_id]-value))

    policy = np.zeros([env.nS, env.nA])

    # produce the optimal policy
    for state_id in range(env.nS):
        action_id = A_state[state_id]
        policy[state_id][action_id] = 1

    print(policy)

    return policy


# Q4:
def policy_evaluation_PI(policy, env, gamma=0.95, theta=0.0001):

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

    # print(V)

    return V, A_state


def policy_improvement(policy,V,A_state,env):

    is_policy_stable = False
    policy_cp = policy

    # produce the optimal policy
    for state_id in range(env.nS):
        action_id = A_state[state_id]
        policy[state_id][action_id] = 1

    if np.allclose(policy,policy_cp):
        is_policy_stable = True

    return policy, is_policy_stable


def policy_iteration(env, gamma=0.95):

    policy = np.ones([env.nS,env.nA])/env.nA
    theta = 0.0001
    V = np.zeros(env.nS)
    is_policy_stable = False

    while not is_policy_stable:

        V, A_state = policy_evaluation_PI(policy,env,gamma,theta)

        policy, is_policy_stable = policy_improvement(policy,V,A_state,env)

    for state_id in range(env.nS):
        action = np.argmax(policy[state_id][:])
        policy[state_id][:] = 0
        policy[state_id][action] = 1

    print(policy)

    return policy


# Q5:
def monte_carlo(env, gamma=0.95, epsilon=0.1, num_episodes=5000):

    # initialize
    Q = np.zeros([env.nS,env.nA])
    Returns = np.zeros([env.nS,env.nA])
    N = np.zeros([env.nS,env.nA])
    state_id = 0
    np.random.seed(42)

    for i_episode in range(num_episodes):

        env.reset()
        done = False
        state_record = np.empty((0,2),dtype=int)
        state_first_visit = np.zeros([env.nS,env.nA])
        reward_record = np.empty((0,1))
        num_step = 0

        # Generate episode
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

            # record the return
            state_record = np.append(state_record,[[state_id,action]],axis=0)
            reward_record = np.append(reward_record,reward*(gamma**num_step))
            state_first_visit[state_id][action] = 1

            # if reward != 0:
            #     print("Great, found goal!")

            state_id = next_state_id
            num_step += 1

        print("Episode finished after {} timesteps".format(num_step))

        # update return table
        for index in range(state_record.shape[0]):
            state_id = state_record[index][0]
            action_id = state_record[index][1]

            Returns_buf_this_ep = 0
            if (state_first_visit[state_id][action_id]):
                Returns_buf_this_ep = np.sum(reward_record[index:])/(gamma**index)
                N[state_id][action_id] += 1 # at most one appearance each episode
                state_first_visit[state_id][action_id] = False

            Returns[state_id][action_id] += Returns_buf_this_ep

            Returns_buf = Returns[state_id][action_id]
            num_apperance = N[state_id][action_id]

            Q[state_id][action_id] = Returns_buf/num_apperance

    print(Q)

    policy = np.zeros([env.nS,env.nA])
    for state_id in range(env.nS):
        for action_id in range(env.nA):
            opt_action = np.argmax(Q[state_id][:])
            if action_id == opt_action:
                policy[state_id][action_id] = 1-epsilon+epsilon/env.nA
            else:
                policy[state_id][action_id] = epsilon/env.nA

    print(policy)

    return Q,policy


# Q6:
def q_learning(env, alpha=0.5, gamma=0.95, epsilon=0.1, num_episodes=500):

    # initialization
    Q = np.zeros([env.nS,env.nA])

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
    # Q1:
    bandits = [bandit_one, bandit_two, bandit_three, bandit_four]
    num_episodes = 100
    for epsilon in [0.3, 0.1, 0.03]:
        Q = learn_bandit_rewards(bandits, epsilon, num_episodes)
        # plt.plot(range(1, num_episodes + 1), Q_max)

    # plt.ylabel('Q_max')
    # plt.xlabel('episodes')
    # plt.legend(['epsilon = 0.3', 'epsilon = 0.1', 'epsilon = 0.03'])
    # plt.show()


    # Q2:
    env = gym.make('FrozenLake-v0').unwrapped
    policy = np.ones([env.nS,env.nA])/env.nA
    policy_evaluation(policy,env)


    # Q3:
    env.reset()
    value_iteration(env)


    # Q4:
    env.reset()
    policy_iteration(env)


    # Q5:
    env.reset()
    monte_carlo(env)


    # Q6:
    env.reset()
    q_learning(env)

    return


if __name__ == "__main__":
    main()