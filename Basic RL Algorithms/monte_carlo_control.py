import gym
import numpy as np


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


def main():
    env = gym.make('FrozenLake-v0').unwrapped
    monte_carlo(env)


if __name__ == "__main__":
    main()