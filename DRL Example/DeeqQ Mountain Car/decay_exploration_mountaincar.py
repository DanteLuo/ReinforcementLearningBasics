import gym
import csv

from deepq_training import train_deepq


def main():

    maximum_steps = 100000
    exploration_start = 1.
    decay_end_steps = 900
    exploration_ends = 0.1
    exploration_frac = (exploration_start-exploration_ends)/maximum_steps*decay_end_steps

    env = gym.make("MountainCar-v0")
    # Enabling layer_norm here is import for parameter space noise!
    act, avg_reward = train_deepq(env, exploration_start=exploration_start, exploration_fraction=exploration_frac,
                                  exploration_final_eps=exploration_ends,max_timesteps=maximum_steps)

    f = open('decay_avg', 'wt')
    try:
        writer = csv.writer(f)
        writer.writerow(avg_reward)
    finally:
        f.close()


if __name__ == '__main__':
    main()
