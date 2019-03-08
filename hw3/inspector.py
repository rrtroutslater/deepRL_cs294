import pickle 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

# consider using a moving average on the reward ... reward itself extremely noisy in plots

def main():
    fnames = [
        '84d1f2bf-5149-4b77-ad60-14791ff9f7e3.pkl',
        '5059ffdb-a8ef-47da-834e-b6af2f66e215.pkl',]
        # 'fa4538ac-2839-456d-8b92-8c4877fd0307.pkl',
        # '68450631-91c3-4561-b45f-3a63a667cd29.pkl',
        # '4aadac89-892c-4339-8728-685a8811b715.pkl']

    labels=['DQN', 'Double DQN']

    with open(fnames[0], 'rb') as f:
        dqn = pd.Series(pickle.load(f))
    dqn_mean = dqn.rolling(10).mean()
    dqn_std = dqn.rolling(10).std()

    with open(fnames[1], 'rb') as f:
        d_dqn = pd.Series(pickle.load(f))
    d_dqn_mean = d_dqn.rolling(10).mean()
    d_dqn_std = d_dqn.rolling(10).std()

    plt.figure()
    plt.plot(dqn_mean[:1500], c='r', label='DQN')
    plt.plot(d_dqn_mean[:1500], c='b', label='Double DQN')
    plt.legend()
    plt.grid()
    plt.title('Mean reward vs # of Training Episodes')
    plt.show()

    plt.figure()
    plt.plot(dqn_std[:1500], c='r', label='DQN')
    plt.plot(d_dqn_std[:1500], c='b', label='Double DQN')
    plt.legend()
    plt.grid()
    # plt.show()


if __name__ == "__main__":
    main()