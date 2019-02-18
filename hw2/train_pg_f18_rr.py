"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Michael Chang and Soroush Nasiriany
implementation completed by Russell Reinhart
"""
import numpy as np 
import tensorflow as tf 
import gym 
import logz
import os
import time
import inspect 
from multiprocessing import Process 


# Utilities 
def build_mlp(input_placeholder, output_size, scope, n_layers, hidden_size, 
        activation=tf.nn.relu, output_activation=None):
    hidden_output = tf.contrib.layers.repeat(
        input_placeholder, n_layers-1, tf.contrib.layers.fully_connected, hidden_size, 
        activation_fn=activation)
            #, scope=scope)

    output_placeholder = tf.layers.dense(
        hidden_output, output_size, activation=output_activation)

    return output_placeholder


def pathlength(path):
    return len(path["reward"])


def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)


# Policy Gradient
class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, 
            estimate_return_args):
        super(Agent, self).__init__()
        self.obs_dim = computation_graph_args['obs_dim']
        self.act_dim = computation_graph_args['act_dim']
        self.discrete = computation_graph_args['discrete']
        self.hidden_size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_match = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_return_args['gamma']
        self.reward_to_go = estimate_return_args['reward_to_go']
        self.nn_baseline = estimate_return_args['nn_baseline']
        self.normalize_advantages = estimate_return_args['normalize_advantages']
    

    def init_tf_sess(self):
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()
        tf.global_variables_initializer().run()


    def define_placeholders(self):
        sy_obs_no = tf.placeholder(shape=[None, self.obs_dim], name='ob', dtype=tf.float32)
        sy_adv_n = tf.placeholder(shape=[None], name='adv', dtype=tf.float32)

        if self.discrete:
            sy_act_n = tf.placeholder(shape=[None], name='ac', dtype=tf.int32)
            return sy_obs_no, sy_act_n, sy_adv_n
        else:
            sy_act_na = tf.placeholder(shape=[None, self.act_dim], name='ac', dtype=tf.float32)
            return sy_obs_no, sy_act_na, sy_adv_n


    def policy_forward_pass(self, sy_obs_no):
        net_out = build_mlp(sy_obs_no, self.act_dim, 'policy', self.n_layers, self.hidden_size)
        if self.discrete:
            sy_logits_na = net_out
            return sy_logits_na
        else:
            sy_mean = net_out
            sy_logstd = tf.Variable(np.random.random(self.act_dim), expected_shape=[self.act_dim], trainable=True, dtype=tf.float32)
            return (sy_mean, sy_logstd)


    def sample_action(self, policy_parameters):
        if self.discrete:
            # tf.random.multinomial(tensor, num_samples) takes (n, num classes), 
            # treates each element as a pdf, and draws num_samples from each
            # logits = tf.nn.log_softmax(policy_parameters)
            sy_sampled_act = tf.multinomial(logits=policy_parameters, num_samples=1)
            sy_sampled_act = tf.reshape(sy_sampled_act, [-1])
        else:
            # x = mu + sigma * z, z ~ N(0, I) -> x sampled from N(mu, sigma)
            sy_mean, sy_logstd = policy_parameters
            z_sample_na = tf.random_normal(tf.shape(sy_mean))
            # sy_sampled_act = sy_mean + tf.exp(sy_logstd) * z_sample_na
            sy_sampled_act = sy_mean + sy_logstd * z_sample_na
        return sy_sampled_act


    def get_log_prob(self, policy_parameters, sy_act_na):
        if self.discrete:
            # log probability under categorical distribution
            sy_logits_na = policy_parameters
            sy_act_na = tf.one_hot(tf.transpose(sy_act_na), self.act_dim)
            sy_logprob_n = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=sy_act_na, logits=sy_logits_na)
            return sy_logprob_n
        else:
            # log probability of given actions under multivariate gaussian with mean 
            # defined by policy
            sy_mean, sy_logstd = policy_parameters
            # sy_z = (sy_act_na - sy_mean) / tf.exp(sy_logstd)
            sy_z = (sy_act_na - sy_mean) / sy_logstd
            sy_logprob_n = -0.5 * tf.reduce_sum(tf.square(sy_z), axis=1)
            return sy_logprob_n


    def build_computation_graph(self):
        self.sy_obs_no, self.sy_act_na, self.sy_adv_n = self.define_placeholders()

        # observation -> distribution over actions 
        self.policy_parameters = self.policy_forward_pass(self.sy_obs_no)

        # sample action from distribution defined by forward pass output
        self.sy_sampled_act = self.sample_action(self.policy_parameters)

        # logprob of actions actually taken by the policy. used in loss function
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_act_na)

        # loss and gradient update
        if self.discrete:
            loss = tf.reduce_mean(tf.multiply(self.sy_logprob_n, self.sy_adv_n))
        else:
            loss = -tf.reduce_mean(tf.multiply(self.sy_logprob_n, self.sy_adv_n))
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        print('act dim:', self.act_dim)
        print('obs dim:', self.obs_dim)
        print('sy_act_na shape:', self.sy_act_na.get_shape())
        print('sy_obs_no shape:', self.sy_obs_no.get_shape())
        print('sy_sampled shape:', self.sy_sampled_act.get_shape())

        # baseline network, trained to predict reward
        if self.nn_baseline:
            self.baseline_prediction = tf.squeeze(build_mlp(
                    self.sy_obs_no,
                    1,
                    "baseline_prediction",
                    n_layers=self.n_layers,
                    hidden_size=self.hidden_size))
            self.sy_target_n = tf.placeholder(shape=[None], name='target', dtype=tf.float32)
            baseline_loss = tf.reduce_sum(tf.square(self.baseline_prediction - self.sy_target_n))
            self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(baseline_loss)


    def sample_trajectories(self, itr, env):
        timesteps_this_batch = 0
        paths = []
        animate_once = False
        while True:
            # animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and self.animate)
            if itr % 20 == 0 and not animate_once:
                animate_this_episode = True
                animate_once = True
            else:
                animate_this_episode = False
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_match:
                break
        return paths, timesteps_this_batch


    def sample_trajectory(self, env, animate_this_episode):
        obs = env.reset()
        obs_list, act_list, rew_list = [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                # env.render()
                time.sleep(0.1)
            obs_list.append(obs)
            o = np.expand_dims(obs, axis=0)
            act = self.sess.run(
                self.sy_sampled_act, 
                feed_dict={self.sy_obs_no : o} )
            a = act
            act = act[0]
            # print(act)
            act_list.append(act)
            obs, rew, done, _ = env.step(act)
            rew_list.append(rew)
            steps += 1

            if done or steps > self.max_path_length:
                break

        path = {"observation": np.array(obs_list, dtype=np.float32),
                "reward": np.array(rew_list, dtype=np.float32),
                "action": np.array(act_list, dtype=np.float32)}
        return path


    def sum_of_rewards(self, re_n):
        n = 0
        for traj in re_n:
            n += traj.shape[0]
        q_n = np.zeros(n)

        # this is probably slower than it needs to be...
        if self.reward_to_go:
            idx = 0
            for traj in re_n:
                for i in range(0, traj.shape[0]):
                    q_n[idx+i] = np.sum(traj[i:]) * self.gamma**i
                idx += traj.shape[0]
        else:
            idx = 0
            for traj in re_n:
                path_reward = 0
                for i in range(0, traj.shape[0]):
                    path_reward += traj[i] * self.gamma**(i-idx)
                q_n[idx:traj.shape[0]] = path_reward                
                idx += traj.shape[0]
        return q_n


    def compute_advantage(self, obs_no, q_n):
        if self.nn_baseline:
            # use network to predict reward-to-go at each teimstep for each trajectory
            # problem 6
            b_n = self.sess.run(self.baseline_prediction,
                # feed_dict={self.sy_obs_no : np.expand_dims(obs_no, axis=0)})
                feed_dict={self.sy_obs_no : obs_no})
            b_n = (b_n - np.mean(q_n)) / np.std(q_n)
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()
        return adv_n


    def estimate_return(self, obs_no, rew_n):
        q_n = self.sum_of_rewards(rew_n)
        adv_n = self.compute_advantage(obs_no, q_n)

        if self.normalize_advantages:
            # empirically, advantage with mu=0, std=1 gives lower training variance
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n))
        return q_n, adv_n


    def update_parameters(self, obs_no, act_na, q_n, adv_n):
        # baseline network parameter update
        if self.nn_baseline:
            # target values for baseline network is normalized reward-to-go
            target_n = (q_n - np.mean(q_n)) / np.std(q_n)
            feed_dict={
                self.sy_obs_no : obs_no,
                self.sy_target_n : target_n}
            self.sess.run(self.baseline_update_op, feed_dict=feed_dict)

        # perform policy update
        feed_dict = {
            self.sy_obs_no : obs_no,
            self.sy_act_na : act_na, 
            self.sy_adv_n : adv_n,}
        self.sess.run(self.update_op, feed_dict=feed_dict)
        return


def train_PG(
    exp_name,
    env_name,
    n_iter,
    gamma,
    min_timesteps_per_batch,
    max_path_length,
    learning_rate,
    reward_to_go,
    animate,
    logdir,
    normalize_advantages,
    nn_baseline,
    seed,
    n_layers,
    size):

    start = time.time()
    setup_logger(logdir, locals()) 

    env = gym.make(env_name)

    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    max_path_length = max_path_length or env.spec.max_episode_steps

    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # initialize agent, build computation graph
    computation_graph_args = {
        'n_layers' : n_layers,
        'obs_dim' : obs_dim,
        'act_dim' : act_dim,
        'discrete' : discrete,
        'size' : size,
        'learning_rate' : learning_rate,
        }

    sample_trajectory_args = {
        'animate' : animate,
        'max_path_length' : max_path_length,
        'min_timesteps_per_batch' : min_timesteps_per_batch,
    }

    estimate_return_args = {
        'gamma' : gamma,
        'reward_to_go' : reward_to_go, 
        'nn_baseline' : nn_baseline,
        'normalize_advantages' : normalize_advantages,
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)
    agent.build_computation_graph()
    agent.init_tf_sess()

    # training loop
    total_timesteps = 0

    for itr in range(n_iter):
        print('**** Iteration %i ****' % itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # build obs, act, rew arrays for PG update by concatenating across paths
        obs_no = np.concatenate([path['observation'] for path in paths])
        act_na = np.concatenate([path['action'] for path in paths])
        rew_n = [path['reward'] for path in paths]
        
        q_n, adv_n = agent.estimate_return(obs_no, rew_n)
        agent.update_parameters(obs_no, act_na, q_n, adv_n)

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=0.995)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--animate', '-an', type=bool, default=True)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                size=args.size
                )
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block 
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()























