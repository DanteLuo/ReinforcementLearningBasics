import gym
import numpy as np
from hw1 import policy_evaluation_PI
from value_iteration import print_value_function, print_policy


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

    # print_value_function(V)
    # print_policy(policy)
    # print(policy)

    print(policy)

    return policy


def main():
    env = gym.make('FrozenLake-v0').unwrapped
    policy_iteration(env)


if __name__ == "__main__":
    main()
