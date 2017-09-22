import numpy as np
import matplotlib.pyplot as plt
import gym


def policy_evaluation(policy, env, gamma=0.95, theta=0.0001):

    env.reset()
    V = np.zeros(env.nS)
    delta = theta+1.

    while delta>=theta:
        delta = 0.

        for state_id in env.P:

            value     = V[state_id]
            new_value = 0.

            for action_id in env.P[state_id]:

                action_value = 0.

                for transition_id in range(len(env.P[state_id][action_id])):

                    transition_prob = env.P[state_id][action_id][transition_id][0]
                    next_state_id   = env.P[state_id][action_id][transition_id][1]
                    reward          = env.P[state_id][action_id][transition_id][2]
                    done            = env.P[state_id][action_id][transition_id][3]

                    if done:
                        action_value += transition_prob*reward
                    else:
                        action_value += transition_prob*(reward+gamma*V[next_state_id])

                new_value += action_value*policy[state_id][action_id]

            V[state_id] = new_value
            delta = max(delta,abs(V[state_id]-value))

    print_value_function(V)

    return V


def print_value_function(v):
    for i in range(4):
        for j in range(4):
            print("%.3f" % v[i*4 + j], end=' ')
        print()
    return


env = gym.make('FrozenLake-v0').unwrapped
env.render()
policy = np.ones([env.nS,env.nA])/env.nA
policy_evaluation(policy,env)


