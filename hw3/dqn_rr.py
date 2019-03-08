import uuid
import time
import pickle
import sys
import gym.spaces
import itertools 
import numpy as np 
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers 
from collections import namedtuple 
from dqn_utils import *

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

class QLearner(object):

    def __init__(
            self,
            env,
            q_func, # input the network 
            optimizer_spec, 
            session,
            exploration=LinearSchedule(1000000, 0.01),
            stopping_criterion=None,
            replay_buffer_size=1000000,
            batch_size=32,
            gamma=0.99,
            learning_starts=50000,
            learning_freq=4,
            frame_history_len=4,
            target_update_freq=10000,
            grad_norm_clipping=10,
            rew_file=None,
            double_q=True,
            lander=False,):
        
        """
        Run Deep Q-Learning Algorithm
        """
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete 

        self.target_update_freq = target_update_freq
        self.optimizer_spec = optimizer_spec
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_freq = learning_freq
        self.learning_starts = learning_starts
        self.stopping_criterion = stopping_criterion
        self.env = env
        self.session = session
        self.exploration = exploration
        self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file
        self.double_q = double_q

        # define observation and action dimensions
        if len(self.env.observation_space.shape) == 1:
            input_shape = self.env.observation_space.shape 
        else:
            img_h, img_w, img_c = self.env.observation_space.shape 
            input_shape = (img_h, img_w, frame_history_len * img_c)
        self.num_actions = self.env.action_space.n

        # placeholders
        self.obs_t_ph = tf.placeholder(
            tf.float32 if lander else tf.uint8, [None] + list(input_shape))
        self.obs_tp1_ph = tf.placeholder(
            tf.float32 if lander else tf.uint8, [None] + list(input_shape))
        self.act_t_ph = tf.placeholder( 
            tf.int32, [None])
        self.rew_t_ph = tf.placeholder(
            tf.float32, [None])
        self.done_mask_ph = tf.placeholder(
            tf.float32, [None])
        
        # casting to float on GPU ensures loewr data transfer times.
        if lander:
            obs_t_float = self.obs_t_ph
            obs_tp1_float = self.obs_tp1_ph
        else:
            obs_t_float = tf.cast(self.obs_t_ph, tf.float32) / 255.0
            obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

        # define q function and target network
        self.q_t = q_func(obs_t_float, self.num_actions, scope='q_act', reuse=False)
        self.q_tp1 = q_func(obs_tp1_float, self.num_actions, scope='q_target', reuse=False)

        if self.double_q:
            print('using double q learning')
            # action selecting network used to select action at time t+1, instead of target network
            self.q_tp1_target = q_func(obs_tp1_float, self.num_actions, scope='q_act', reuse=True)
            best_act_tp1 = tf.argmax(self.q_tp1_target, axis=1)
            best_q_tp1 = tf.reduce_sum(self.q_tp1 * tf.one_hot(best_act_tp1, self.num_actions), axis=1)
        else:
            print('NOT using double q learning')
            # target network selects action at t+1
            best_q_tp1 = tf.reduce_max(self.q_tp1, axis=1)  # assume greedy action selection at next time

        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_act')

        self.best_act = tf.argmax(self.q_t, axis=1)     # for greedy action selection
        q_t_taken = tf.reduce_sum(tf.one_hot(self.act_t_ph, self.num_actions) * self.q_t, axis=1)

        # bellman error: r_t + gamma*Q_t+1 - Q_t
        y_t = self.rew_t_ph + self.gamma * tf.multiply((1.0 - self.done_mask_ph), best_q_tp1) - \
            q_t_taken
        self.total_error = huber_loss(y_t)
        # MSE
        # self.total_error = tf.losses.mean_squared_error(
        #     self.rew_t_ph + self.gamma * tf.multiply((1.0 - self.done_mask_ph), best_q_tp1),
        #     q_t_taken)

        # optimization
        self.learning_rate = tf.placeholder(tf.float32, (), name='learning_rate')
        optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
        self.train_fn = minimize_and_clip(
            optimizer, self.total_error,
            var_list=q_func_vars, clip_val=grad_norm_clipping)

        # periodically copy Q network to target Q network
        update_target_fn = [] 
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target')
        for var, var_target in zip(sorted(q_func_vars, key=lambda v:v.name),
                                    sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

        # construct replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
        self.replay_buffer_idx = None

        # run environment
        self.model_initialized = False
        self.num_param_updates = 0
        self.mean_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.last_obs = self.env.reset()
        self.log_every_n_steps = 10000

        self.start_time = None
        self.t = 0

    def stopping_criterion_met(self):
        return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

    def step_env(self):
        # 1) get input from the replay buffer
        idx = self.replay_buffer.store_frame(self.last_obs) # store obs in buffer, get index of its location
        obs_in = self.replay_buffer.encode_recent_observation()
        obs_in = np.expand_dims(obs_in, axis=0)

        # 2) epsilon greedy action selection
        if not self.model_initialized or random.random() < self.exploration.value(self.t):
            act = np.random.randint(0, self.num_actions)
        else:
            act = self.session.run(
                self.best_act,
                feed_dict={self.obs_t_ph : obs_in})[0]

        # 3) step environemnt, store transition
        obs, reward, done, info = self.env.step(act)

        if done:
            self.last_obs = self.env.reset()
        else:
            self.last_obs = obs

        self.replay_buffer.store_effect(idx, act, reward, done)
        return

    def update_model(self):
        # experience replay and network training

        if (self.t > self.learning_starts and \
            self.t % self.learning_freq == 0 and \
            self.replay_buffer.can_sample(self.batch_size)):

            # use replay buffer to sample batch of transitions
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(self.batch_size)

            # initialize model
            if not self.model_initialized:
                initialize_interdependent_variables(self.session, tf.global_variables(),
                    feed_dict = {self.obs_t_ph : obs_batch,
                                self.obs_tp1_ph : next_obs_batch})
                self.model_initialized = True

            # model training
            self.session.run(
                self.train_fn,
                feed_dict={
                    self.rew_t_ph : rew_batch,
                    self.obs_t_ph : obs_batch,
                    self.obs_tp1_ph : next_obs_batch,
                    self.act_t_ph : act_batch,
                    self.done_mask_ph : done_mask,
                    self.learning_rate : self.optimizer_spec.lr_schedule.value(self.t)})
            
            # periodically update target network (not every time) to reduce variance
            if self.num_param_updates % self.target_update_freq == 0:
                self.session.run(
                    self.update_target_fn)
            
            self.num_param_updates += 1
        self.t += 1

    def log_progress(self):
        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards() 

        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)
        
        if self.t % self.log_every_n_steps == 0 and self.model_initialized:
            print('timestep %d' % (self.t,))
            print('mean reward (100 episodes) %f' % self.mean_episode_reward)
            print('best mean reward %f' % self.best_mean_episode_reward)
            print('episodes %d' % len(episode_rewards))
            print('exploration %f' % self.exploration.value(self.t))
            print('learning_rate %f' % self.optimizer_spec.lr_schedule.value(self.t))
            if self.start_time is not None:
                print('running time %f' % ((time.time() - self.start_time) / 60.0))
            
            self.start_time = time.time()

            sys.stdout.flush()

            with open(self.rew_file, 'wb') as f:
                pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)


def learn(*args, **kwargs):
    alg = QLearner(*args, **kwargs)
    while not alg.stopping_criterion_met():
        alg.step_env()
        alg.update_model()
        alg.log_progress()


# """
#     lesson learned
#
#     - q learning requires experience replay
#         - have external object to store/access samples (makes for cleaner graph implementation)
#
#     - q learning requires complex learning rate parameters etc
#         - have external parameter manager that can adjust learning rate as a function of training iteration
#
#     - huber loss 
#         - less sensitive to outliers than square error loss
#         - differentiable at 0
#         - approximately absolute error that becomes quadratic for small errors    
#         - supposedly works better but seems to work worse (for atari ram at least)
#
#     - training:
#         - have a "simple" version of the problem on which to test algorithm for training
# """





























