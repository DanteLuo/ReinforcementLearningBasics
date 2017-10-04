import numpy as np
import frozen_lakes
import gym
import matplotlib.pyplot as plt
from queue import PriorityQueue
import argparse as argp


# Q1:
def dyna_q_learning(env, num_episodes=30, num_planning=50, epsilon=0.1,
                    alpha=0.1, gamma=0.95, kappa=None):

    Q = np.zeros([env.nS,env.nA])
    policy = np.zeros([env.nS,env.nA])
    env_model = {}
    Q_max = np.zeros(num_episodes)
    np.random.seed(22)
    env.seed(22)

    for i_episode in range(num_episodes):

        env.reset()
        state_id = 0 # how to initialize in general case, where no starting state?
        num_step = 0
        done = False

        while not done:

            # epsilon-soft
            randm_num = np.random.rand()

            if randm_num > epsilon:
                action_set = np.argwhere(Q[state_id][:] == np.max(Q[state_id][:]))
                index_rand = np.random.randint(0,len(action_set))
                action = int(action_set[index_rand])
            else:
                action = np.random.randint(env.nA)

            next_state_id, reward, done, info = env.step(action)

            # if next_state_id == 19:
            #     print(reward,action,reward)

            # update Q
            Q[state_id][action] += alpha*(reward+gamma*np.max(Q[next_state_id][:])-Q[state_id][action])

            # model learning
            if (state_id,action) not in env_model.keys():
                env_model[(state_id,action)] = [{},1,i_episode]
                env_model[(state_id, action)][0][next_state_id] = [1, reward]
            else:
                env_model[(state_id,action)][1] += 1  # store the total visit number of (S,A)
                env_model[(state_id,action)][2] = i_episode

                if next_state_id not in env_model[(state_id,action)][0].keys():
                    env_model[(state_id, action)][0][next_state_id] = [1,reward]
                else:
                    N_next_state = env_model[(state_id,action)][0][next_state_id][0]
                    reward_old = env_model[(state_id,action)][0][next_state_id][1]  # the reward record
                    env_model[(state_id,action)][0][next_state_id][0] += 1
                    # average the reward getting
                    env_model[(state_id,action)][0][next_state_id][1] = (N_next_state*reward_old + reward)\
                                                                        /(N_next_state+1)

            # model planning
            for i_planning in range(num_planning):
                randm_state_id_planning = np.random.randint(len(list(env_model)))
                state_id_planning = list(env_model)[randm_state_id_planning][0]
                action_planning = list(env_model)[randm_state_id_planning][1]
                delta_time = i_episode-env_model[(state_id_planning,action_planning)][2]

                # number of visit of (S,A)
                N_S_A = env_model[list(env_model)[randm_state_id_planning]][1]
                # state_transition is a dict with keys of s' and [0] for N_sub [1] for Pss'
                state_transition = env_model[list(env_model)[randm_state_id_planning]][0]
                next_state_set = list(state_transition)

                # Q(S,A) -= alpha(expected_Q-Q)
                expected_Q = 0
                for next_state_id_planning in next_state_set:
                    state_transition_prob = state_transition[next_state_id_planning][0]/N_S_A
                    reward_planning = state_transition[next_state_id_planning][1]
                    if kappa:
                        reward_planning += kappa*np.sqrt(delta_time)
                    Q_planning = np.max(Q[next_state_id_planning][:])
                    expected_Q += state_transition_prob*(reward_planning+gamma*Q_planning)

                Q[state_id_planning][action_planning] += alpha*(expected_Q-Q[state_id_planning][action_planning])

            Q_max[i_episode] = np.max(Q[0][:])
            state_id = next_state_id
            num_step += 1

        print("Episode finished after {} timesteps".format(num_step))

    # retrieve the policy
    for state_id in range(env.nS):
        action = np.argmax(Q[state_id][:])
        policy[state_id][action] = 1

    # plot the Q[0] change
    # plt.plot(range(1, num_episodes + 1), Q_max)
    # plt.ylabel('Q_max')
    # plt.xlabel('episodes')
    # plt.show()
    # print(Q)
    print(policy)

    return policy


# Q2:
def dyna_q_learning_comp_helper(env_one, env_two, num_timesteps=6000, num_planning=50, epsilon=0.1,
                                alpha=0.1, gamma=0.95, kappa=None):

    Q = np.zeros([env_one.nS,env_one.nA])
    env_model = {}
    accumulative_reward_record = np.zeros(num_timesteps)
    num_timesteps_half = int(num_timesteps/2)
    accumulative_reward = 0
    total_num_step = 0
    num_episodes = 0

    for i_num_env in range(2):
        sub_total_num_step = 0
        if i_num_env == 0:
            env = env_one
        else:
            env = env_two

        while sub_total_num_step < num_timesteps_half:

            env.reset()
            state_id = 0  # how to initialize in general case, where no starting state?
            num_step = 0
            done = False
            num_episodes += 1

            while not done and sub_total_num_step < num_timesteps_half:

                # epsilon-soft
                randm_num = np.random.rand()

                if randm_num > epsilon:
                    action_set = np.argwhere(Q[state_id][:] == np.max(Q[state_id][:]))
                    index_rand = np.random.randint(0,len(action_set))
                    action = int(action_set[index_rand])
                else:
                    action = np.random.randint(env.nA)

                next_state_id, reward, done, info = env.step(action)

                # update Q
                Q[state_id][action] += alpha*(reward+gamma*np.max(Q[next_state_id][:])-Q[state_id][action])
                accumulative_reward += reward
                accumulative_reward_record[total_num_step] = accumulative_reward

                # model learning and update model time
                if (state_id,action) not in env_model.keys():
                    env_model[(state_id,action)] = [{},1,num_episodes]
                    env_model[(state_id, action)][0][next_state_id] = [1, reward]
                else:
                    env_model[(state_id,action)][1] += 1  # store the total visit number of (S,A)
                    env_model[(state_id, action)][2] = num_episodes

                    if next_state_id not in env_model[(state_id,action)][0].keys():
                        env_model[(state_id, action)][0][next_state_id] = [1,reward]
                    else:
                        N_next_state = env_model[(state_id,action)][0][next_state_id][0]
                        reward_old = env_model[(state_id,action)][0][next_state_id][1]  # the reward record
                        env_model[(state_id,action)][0][next_state_id][0] += 1
                        # average the reward getting
                        env_model[(state_id,action)][0][next_state_id][1] = (N_next_state*reward_old + reward)\
                                                                            /(N_next_state+1)

                # model planning
                for i_planning in range(int(num_planning)):
                    randm_state_id_planning = np.random.randint(len(list(env_model)))
                    state_id_planning = list(env_model)[randm_state_id_planning][0]
                    action_planning = list(env_model)[randm_state_id_planning][1]
                    delta_time = num_episodes-env_model[(state_id_planning,action_planning)][2]

                    # number of visit of (S,A)
                    N_S_A = env_model[list(env_model)[randm_state_id_planning]][1]
                    # state_transition is a dict with keys of s' and [0] for N_sub [1] for Pss'
                    state_transition = env_model[list(env_model)[randm_state_id_planning]][0]
                    next_state_set = list(state_transition)

                    # Q(S,A) -= alpha(expected_Q-Q)
                    expected_Q = 0
                    for next_state_id_planning in next_state_set:
                        state_transition_prob = state_transition[next_state_id_planning][0]/N_S_A
                        reward_planning = state_transition[next_state_id_planning][1]
                        if kappa:
                            reward_planning += kappa*np.sqrt(delta_time)
                        Q_planning = np.max(Q[next_state_id_planning][:])
                        expected_Q += state_transition_prob*(reward_planning+gamma*Q_planning)

                    Q[state_id_planning][action_planning] += alpha*(expected_Q-Q[state_id_planning][action_planning])

                state_id = next_state_id
                total_num_step += 1
                num_step += 1
                sub_total_num_step += 1

            print("Episode finished after {} timesteps".format(num_step))

    return accumulative_reward_record


def dyna_q_learning_comp(env, env_two, num_averaged = 10, num_timesteps=6000, num_planning=50, epsilon=0.1,
                         alpha=0.1, gamma=0.95, kappa=0.001):

    avg_accumulative_reward_dynaq = np.zeros(num_timesteps)
    avg_accumulative_reward_dynaqp = np.zeros(num_timesteps)

    np.random.seed(42)
    for i_num_averaged in range(num_averaged):

        # generate random seed
        randm_seed = np.random.randint(1000)
        np.random.seed(randm_seed)
        env.seed(randm_seed)
        env_two.seed(randm_seed)

        # run dynaq
        avg_accumulative_reward_dynaq_buf = dyna_q_learning_comp_helper(env, env_two, num_timesteps,
                                                                    num_planning, epsilon,
                                                                    alpha, gamma)
        avg_accumulative_reward_dynaq = (avg_accumulative_reward_dynaq*i_num_averaged
                                         +avg_accumulative_reward_dynaq_buf)/(i_num_averaged+1)

        # run dynaq+
        avg_accumulative_reward_dynaqp_buf = dyna_q_learning_comp_helper(env, env_two, num_timesteps,
                                                                     num_planning, epsilon, alpha, gamma, kappa)
        avg_accumulative_reward_dynaqp = (avg_accumulative_reward_dynaqp * i_num_averaged
                                         + avg_accumulative_reward_dynaqp_buf) / (i_num_averaged + 1)


    plt.plot(range(num_timesteps), avg_accumulative_reward_dynaq,'r')
    plt.plot(range(num_timesteps), avg_accumulative_reward_dynaqp,'b')
    plt.plot(3000*np.ones(20),np.linspace(-120,10,20),'g--')
    plt.ylabel('Cumulative reward')
    plt.xlabel('Time steps')
    plt.legend(['Dyna-Q', 'Dyna-Q+'])
    plt.show()


# Q3:
def prioritized_sweeping(env, num_episodes=30, num_planning=50, epsilon=0.1,
                         alpha=0.1, theta=0.1, gamma=0.95):
    Q = np.zeros([env.nS, env.nA])
    policy = np.zeros([env.nS, env.nA])
    env_model = {}
    accumulative_reward_record = np.zeros(num_episodes*20)
    accumulative_reward = 0
    PQueue = PriorityQueue()
    total_num_step = 0
    # for breaking tie by preferring the oldest one
    num_PQueue = 0
    np.random.seed(22)
    env.seed(22)

    for i_episode in range(num_episodes):

        env.reset()
        state_id = 0  # how to initialize in general case, where no starting state?
        num_step = 0
        done = False

        while not done:

            # epsilon-soft
            randm_num = np.random.rand()

            if randm_num > epsilon:
                action_set = np.argwhere(Q[state_id][:] == np.max(Q[state_id][:]))
                index_rand = np.random.randint(0, len(action_set))
                action = int(action_set[index_rand])
            else:
                action = np.random.randint(env.nA)

            next_state_id, reward, done, info = env.step(action)

            # update PQueue
            P = abs(reward + gamma * np.max(Q[next_state_id][:]) - Q[state_id][action])
            if P > theta:
                PQueue.put(((-P,num_PQueue),(state_id,action)))
                num_PQueue += 1

            accumulative_reward += reward
            if total_num_step < num_episodes*20:
                accumulative_reward_record[total_num_step] = accumulative_reward

            # model learning
            if (state_id, action) not in env_model.keys():
                env_model[(state_id, action)] = [{}, 1]
                env_model[(state_id, action)][0][next_state_id] = [1, reward]
            else:
                env_model[(state_id, action)][1] += 1  # store the total visit number of (S,A)

                if next_state_id not in env_model[(state_id, action)][0].keys():
                    env_model[(state_id, action)][0][next_state_id] = [1, reward]
                else:
                    N_next_state = env_model[(state_id, action)][0][next_state_id][0]
                    reward_old = env_model[(state_id, action)][0][next_state_id][1]  # the reward record
                    env_model[(state_id, action)][0][next_state_id][0] += 1
                    # average the reward getting
                    env_model[(state_id, action)][0][next_state_id][1] = (N_next_state * reward_old + reward)\
                                                                         /(N_next_state + 1)

            # model planning
            num_of_planning_steps = num_planning
            while not PQueue.empty() and num_of_planning_steps > 0:
                state_action_id_planning = PQueue.get()[1]
                state_id_planning = state_action_id_planning[0]
                action_planning = state_action_id_planning[1]

                # number of visit of (S,A)
                N_S_A = env_model[state_action_id_planning][1]
                # state_transition is a dict with keys of s' and [0] for N_sub [1] for Pss'
                state_transition = env_model[state_action_id_planning][0]
                next_state_set = list(state_transition)

                # Q(S,A) -= alpha(expected_Q-Q)
                expected_Q = 0
                for next_state_id_planning in next_state_set:
                    state_transition_prob = state_transition[next_state_id_planning][0] / N_S_A
                    reward_planning = state_transition[next_state_id_planning][1]
                    Q_planning = np.max(Q[next_state_id_planning][:])
                    expected_Q += state_transition_prob * (reward_planning + gamma * Q_planning)

                Q[state_id_planning][action_planning] += alpha * (expected_Q - Q[state_id_planning][action_planning])

                # update PQueue who leads to current state
                # for predictor_state in range(env.nS):
                #     for predictor_action in range(env.nA):
                #         if state_predictor[state_id_planning][predictor_state][predictor_action]:
                #             N_PQ = env_model[(predictor_state,predictor_action)][1]
                #             N_state_PQ = env_model[(predictor_state,predictor_action)][0][state_id_planning][0]
                #             reward_PQ = env_model[(predictor_state,predictor_action)][0][state_id_planning][1]
                #             Q_PQ = np.max(Q[state_id_planning][:])
                #             state_transition_prob = N_state_PQ/N_PQ
                #             P = abs(state_transition_prob*(reward_PQ+gamma*Q_PQ)-Q[predictor_state][predictor_action])
                #             if P > theta:
                #                 PQueue.put((-P,(predictor_state,predictor_action)))

                for state_action_predictor in list(env_model):
                    if state_id_planning in list(env_model[state_action_predictor][0]):
                        state_PQ = state_action_predictor[0]
                        action_PQ = state_action_predictor[1]
                        N_PQ = env_model[state_action_predictor][1]
                        N_state_PQ = env_model[state_action_predictor][0][state_id_planning][0]
                        reward_PQ = env_model[state_action_predictor][0][state_id_planning][1]
                        Q_PQ = np.max(Q[state_id_planning][:])
                        state_transition_prob = N_state_PQ/N_PQ
                        P = abs(state_transition_prob*(reward_PQ+gamma*Q_PQ)-Q[state_PQ][action_PQ])
                        if P > theta:
                            PQueue.put(((-P,num_PQueue), (state_PQ, action_PQ)))
                            num_PQueue += 1

                num_of_planning_steps -= 1

            state_id = next_state_id
            num_step += 1
            total_num_step += 1

        print("Episode finished after {} timesteps".format(num_step))

    # retrieve the policy
    for state_id in range(env.nS):
        action = np.argmax(Q[state_id][:])
        policy[state_id][action] = 1

    # plot the cumulative reward
    plt.plot(range(1, num_episodes*20+1), accumulative_reward_record)
    plt.ylabel('Cumulative reward')
    plt.xlabel('Time step')
    plt.show()
    print(Q)
    print(policy)

    return policy


def main():

    parser = argp.ArgumentParser(description='Please give a question number.')
    parser.add_argument('-q', dest='question', type=int, default=5,
                        help='1 for Dyna-Q.'
                             '2 for Dyna-Q+. '
                             '3 for comparing Dyna-Q and Dyna-Q+'
                             '4 for prioritized sweeping'
                             '5 for running all the questions and is default')
    cmdline = parser.parse_args()

    question = cmdline.question

    env = gym.make('FrozenLakeLarge-v0').unwrapped

    # Q1
    if question == 1 or question == 5:
        policy_one = dyna_q_learning(env,num_episodes=350,kappa=None)

    if question == 2 or question == 5:
        policy_two = dyna_q_learning(env,num_episodes=350,kappa=.0001)

    # Q2
    if question == 3 or question == 5:
        env_two = gym.make('FrozenLakeLargeShiftedIce-v0').unwrapped
        dyna_q_learning_comp(env,env_two,num_averaged=10,kappa=0.001)

    # Q3
    if question == 4 or question == 5:
        policy_three = prioritized_sweeping(env,num_episodes=400,theta=0.1)


if __name__ == '__main__':
    main()
