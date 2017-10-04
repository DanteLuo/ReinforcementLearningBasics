import numpy as np
from queue import PriorityQueue
import matplotlib.pyplot as plt

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


# G = np.ones([3,4])
# G[1][0] = 0

# test = np.sum(G[0][0][:])
#
# print(G)
#
# print(test)

# print(np.min(G[1][:]))


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

# for i in range(50):
#     print(np.random.uniform(0.0,3.0,1))

# Q = np.zeros([16,4])
# # for i in range(16):
# #     print(np.max(Q[i][:]))
#
# # print(Q[:][4])
# state_id = 4
# action_set = np.argwhere(Q[state_id][:] == np.max(Q[state_id][:]))
#
# print(action_set)

# testdict = {}
# testdict[(1,2)] = [{},20]
# testdict[(1,2)][0][1] = (1,2)

# print(testdict[(1,2)][0][1][1])

# print(len(list(testdict)))

# for i in range(100):
#     print(np.random.randint(10))

# if list(testdict)[0] == 1:
#     print('yes!')

# for i in list(testdict):
#     print(i[0])

# kap = None
# if kap:
#     print('no.')
# else:
#     print('yes!')

# myqueue = PriorityQueue()
# myqueue.put(((3,1),(4,2)))
# myqueue.put(((1,3),(2,2)))
#
# print(myqueue.get())
# print(myqueue.qsize())
# print(myqueue.get())
# print(myqueue.qsize())


# N = np.zeros([3,3,2])
# N[1][2][:] = 1
# print(N[1][:][:])
# print(N)

# test = {}
#
# test[1] = [{},1]
#
# if test[1][0][1]:
#     print('yes!')
# else:
#     print('no!')

# test = np.zeros(100)
# test2 = np.zeros(200)
# test3 = []
# test3 = np.concatenate((test,test2))
#
# print(test2-1)

# plt.plot(3000*np.ones(20),np.linspace(-120,10,20),'g--')
# plt.show()

# class test:
#     def __init__(self,ns,na):
#         self.ns = ns
#         self.na = na
#
#
# test_set = test(4,4)
# print(test_set.ns)

model = [[ 1,  0,  0,  0, 0],
         [ 0, -1,  0, -1, 0],
         [ 0,  0,  0,  0, 0],
         [ 0,  0, -1, -1, 0],
         [ 0,  0,  0,  0, 2]]

print(model[3][2])
