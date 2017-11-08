import gym
import csv

from deepq_training import train_deepq


def main():

    maximum_steps = 100000
    exploration_rate = 0.1

    env = gym.make("MountainCar-v0")
    # Enabling layer_norm here is import for parameter space noise!
    act, avg_reward = train_deepq(env,exploration_start=exploration_rate,exploration_fraction=0,
                                  exploration_final_eps=exploration_rate, max_timesteps=maximum_steps)

    f = open('constant_avg', 'wt')
    try:
        writer = csv.writer(f)
        writer.writerow(avg_reward)
    finally:
        f.close()


if __name__ == '__main__':
    main()
