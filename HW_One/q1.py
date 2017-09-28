import numpy as np
import matplotlib.pyplot as plt

def bandit_1():
    return 8

def bandit_2():
    a=np.random.random()
    if a>0.88:
        return 100
    else:
        return 0

def bandit_3():
    return np.random.uniform(-10, 35+1)

def bandit_4():
    a=np.random.random()
    if a<=1/3:
        return 0
    elif a<=2/3:
        return 20
    else:
        return np.random.randint(8, 19)

bandits = [bandit_1,bandit_2,bandit_3,bandit_4]

def learn_bandit_rewards(bandits, epsilon, num_episodes):
     np.random.seed(42)
     Q = np.zeros(4)
     N=np.zeros(4)
     Q_max=[]
     for i_episode in range(num_episodes):
#         for t in range(10000):
         a=np.random.random()
         if a<=epsilon:
             action=np.random.randint(0,4)
             reward=bandits[action]()
         else:
             action=np.argmax(Q)
             reward=bandits[action]()
         N[action]=N[action]+1
         Q[action]=Q[action]+1/N[action]*(reward-Q[action])
         Q_max.append(np.max(Q))
     return Q_max
            
if __name__=="__main__":
    epsilon=[0.3,0.1,0.03]
    num_episodes=100
    Q_max =np.zeros(num_episodes)
    for i in [0,1,2]:
        Q_max=learn_bandit_rewards(bandits, epsilon[i], num_episodes)
        plt.plot(range(num_episodes),Q_max,linestyle='-')
    plt.xlabel('epsilon')
    plt.ylabel('Q_max')
    plt.legend(['epsilon=0.3','epsilon=0.1','epsilon=0.03'])
    plt.show()            
                
        