import os
import gym
import tensorflow as tf 
from BehaviorCloner import *

"""
TODO:
-test dagger
-move network into function
-implement a lstm-based
-generate loss plots/screenshots for report
-when finished, upload to github
"""

envname = 'Hopper-v2'
expert_policy = 'Hopper-v1.pkl'
session = tf.Session()
obs_dim = 11
act_dim = 3

# ---- behavior cloning ---- #
bc = BehaviorCloner(envname, session, obs_dim, act_dim)
bc_fname = 'hopper_behavior_clone'
expert_data_fname = bc.run_expert_policy(expert_policy, render=False, num_rollout=8)

bc.train_behavior_clone(expert_data_fname, restore=False, iter_train=8000, 
    save_model=True, model_fname=bc_fname)

_, bc_reward = bc.run_novice_policy(restore=True, model_fname=bc_fname, save_data=False)
session.close()
bc_reward = np.array(bc_reward)
print(np.mean(bc_reward))
print(np.var(bc_reward))
plt.figure()
plt.plot(bc_reward)
plt.title('reward vs. iteration')
plt.show()
# -------------------------- #

# ---- dagger training ----- #
# d = BehaviorCloner(envname, session, obs_dim, act_dim)
# d_fname = 'hopper_dagger'
# _, loss_list = d.train_dagger(
#     expert_policy, restore=False, model_fname=d_fname, num_epoch=8, 
#     num_expert_rollout=3, iter_test_novice=500, plot_loss=False)

# plt.figure()
# plt.plot(loss_list)
# plt.title('loss vs. iteration')
# plt.show()

# _, reward_list = d.run_novice_policy(restore=True, model_fname=d_fname)
# d_reward = np.array(reward_list)
# print(np.mean(d_reward))
# print(np.var(d_reward))

# plt.figure()
# plt.plot(reward_list)
# plt.title('reward vs. iteration')
# plt.show()
# -------------------------- #
session.close()














