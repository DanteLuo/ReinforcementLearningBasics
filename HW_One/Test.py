import numpy as np

# test = {}
#
# if 'Tom' in test:
#     print('Array is not empty.')
# else:
#     print('Dict is empty!')
#
# test[(1,1)] = (1,1)
#
# print(test[(1,1)][0])
#
# buf = test[(1,1)]
# test[(1,1)] = (buf[0],buf[1]+1)
#
# print(test)


G = np.ones([3,4])
G[1][0] = 0

# test = np.sum(G[0][0][:])
#
# print(G)
#
# print(test)

print(np.min(G[1][:]))


# state_record = np.empty((0,3))
# state_record = np.append(state_record,[[1,2,3]],axis=0)
# state_record = np.append(state_record,[[2,3,4]],axis=0)
#
# for i in range(5):
#     print(i,"this is ")
#
# print(state_record)
#
# print(2**8)


# state_first_visit = np.zeros([16,10,2])
#
# print(state_first_visit)
# print(state_first_visit[:][1][1].shape)


# import gym
# env = gym.make('FrozenLake-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     print(observation)
#     for t in range(100):
#         env.render()
#         # print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             # print("Episode finished after {} timesteps".format(t+1))
#             break


