import gym
import numpy as np


def print_value_function(v):
    for i in range(4):
        for j in range(4):
            print("%.3f" % v[i*4 + j], end=' ')
        print()
    return


def print_policy(policy):
    characters = ['↑','←', '↓', '→']
    for i in range(4):
        for j in range(4):
            print(characters[np.argmax(policy[i*4 + j])], end=' ')
        print()
    return


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

    print_value_function(V)
    policy = np.zeros([env.nS, env.nA])

    # produce the optimal policy
    for state_id in range(env.nS):
        action_id = A_state[state_id]
        policy[state_id][action_id] = 1

    # print(A_state)
    # print_policy(policy)
    # print(Q)

    print(policy)

    return policy


def main():
    env = gym.make('FrozenLake-v0').unwrapped
    value_iteration(env)


if __name__ == "__main__":
    main()