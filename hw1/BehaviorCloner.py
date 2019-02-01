import os
import gym
# import load_policy
from load_policy import *
from run_expert import *
from tf_util import *
import matplotlib.pyplot as plt

class BehaviorCloner():

    def __init__(self,
            envname,
            session,
            obs_dim,
            act_dim,
            learning_rate=0.0001,
            name='policy_emulator',
            expert_policy_dir='./expert_policy/',
            novice_policy_dir='./novice_policy/',
            expert_data_dir='./expert_data/',
            novice_data_dir='./novice_data/'):
        """

        """
        self.envname = envname
        self.session = session
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.learning_rate = learning_rate
        self.expert_policy_dir = expert_policy_dir
        self.expert_data_dir = expert_data_dir
        self.novice_policy_dir = novice_policy_dir
        self.novice_data_dir = novice_data_dir 

        if not os.path.isdir(self.expert_policy_dir):
            os.mkdir(self.expert_policy_dir)
        if not os.path.isdir(self.expert_data_dir):
            os.mkdir(self.expert_data_dir)
        if not os.path.isdir(self.novice_policy_dir):
            os.mkdir(self.novice_policy_dir)
        if not os.path.isdir(self.novice_data_dir):
            os.mkdir(self.novice_data_dir)

        if self.obs_dim > 11:
            self.hidden_dim = 3 * self.obs_dim
        else:
            self.hidden_dim = 11 * self.obs_dim

        self.layer_sizes = {
            'h1_d': 64,
            'h2_d': 128,
            'h3_d': 128,
            'h4_d': 64,}

        # define the model
        with tf.variable_scope(name) as scope:
            self.obs_train = tf.placeholder(tf.float32, [None, self.obs_dim], 'obs_train')
            self.act_train = tf.placeholder(tf.float32, [None, self.act_dim], 'act_train')
            self.is_train = True
            self.scope_name = scope.name

            # a few fully-connected layers with relu activation and batch normalization
            fc_1 = tf.layers.dense(
                self.obs_train, self.layer_sizes['h1_d'], tf.nn.relu, trainable=self.is_train)
            bn_1 = tf.layers.batch_normalization(fc_1)
            fc_2 = tf.layers.dense(
                bn_1, self.layer_sizes['h2_d'], tf.nn.relu, trainable=self.is_train)
            bn_2 = tf.layers.batch_normalization(fc_2)
            fc_3 = tf.layers.dense(
                bn_2, self.layer_sizes['h3_d'], tf.nn.tanh, trainable=self.is_train)

            self.out = tf.layers.dense(
                fc_3, self.act_dim, trainable=self.is_train, name='out')

            self.loss = tf.losses.mean_squared_error(self.out, self.act_train)

            with tf.variable_scope('training'):
                opt = tf.train.AdamOptimizer(self.learning_rate)
                self.train_step = opt.minimize(self.loss)

            self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope_name)
            self.init_vars = tf.variables_initializer(self.variables)
            self.saver = tf.train.Saver(self.variables)
   
    def run_novice_policy(self,
            restore=False,
            num_iter=1000,
            num_rollout=3,
            render=True,
            save_data=True,
            model_fname=None,
            novice_data_fname=None,
            plot=True):
        """ 
        """        
        if restore:
            self.restore_model(model_fname)

        env = gym.make(self.envname)
        # max_steps = min(num_iter, env.spec.timestep_limit)
        if num_iter < env.spec.timestep_limit:
            max_steps = num_iter
        else:
            max_steps = env.spec.timestep_limit


        obs_list = []
        act_list = []
        reward_list = []

        # run simulation, acquire observation and corresponding action data
        for i in range(0, num_rollout):
            obs = env.reset()
            done = False
            steps = 0

            while not done:
                obs = np.reshape(obs, (1, obs.shape[0]))
                feed_dict = {self.obs_train : obs}
                act = self.session.run(self.out, feed_dict=feed_dict)
                obs_list.append(obs)
                act_list.append(act)
                obs, r, _, _ = env.step(act)
                reward_list.append(r)
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    done = True
        
        novice_data = {
            'observations': np.array(obs_list).reshape(len(obs_list), self.obs_dim),
            'actions': np.array(act_list).reshape(len(act_list), self.act_dim)}

        if novice_data_fname is None:
            novice_data_fname = self.envname + '_novice.pkl'

        if save_data:
            with open(self.novice_data_dir + novice_data_fname, 'wb') as f:
                pickle.dump(novice_data, f, pickle.HIGHEST_PROTOCOL)
            print('novice policy data saved at:', self.novice_data_dir + novice_data_fname)

        if plot:
            # plot action time-histories
            num_subplots = novice_data['actions'].shape[1]
            num_timesteps = novice_data['actions'].shape[0]
            fig = plt.figure()
            for i in range(1, num_subplots + 1):
                ax = fig.add_subplot(num_subplots, 1, i)
                ax.plot(np.linspace(0, num_timesteps, num_timesteps), novice_data['actions'][:num_timesteps, i-1])
            plt.show()

        return novice_data_fname, reward_list


    def run_expert_policy(self,
            expert_policy_file,
            render=True,
            num_rollout=5,
            max_timesteps=1000,
            dagger=False,
            expert_data_fname=None,
            novice_data_fname=None):
        """ 
        # TODO
        """

        if dagger and expert_data_fname is None or novice_data_fname is None:
            print('must provide previous expert data, and new novice data for dagger')

        if expert_data_fname is None:       
            expert_data_fname = self.envname + str(num_rollout * max_timesteps) + '_expert.pkl'

        # load data from novice for dataset aggregation
        if dagger:
            print("****", novice_data_fname)
            novice_data = self.load_novice_data(novice_data_fname)
            training_obs = novice_data['observations']
            max_steps = training_obs.shape[0]
            expert_data = self.load_expert_data(expert_data_fname)

        # load expert policy - different framework from 
        print('loading and building expert policy function...')
        policy_fn = load_policy.load_policy(self.expert_policy_dir + expert_policy_file)
        print('loaded and built')

        with tf.Session(): 
            tf_util.initialize()
            env = gym.make(self.envname)
            observations = []
            actions = []

            if not dagger:
                # max_steps = np.min(max_timesteps, env.spec.timestep_limit)
                if max_timesteps < env.spec.timestep_limit:
                    max_steps = max_timesteps
                else:
                    max_steps = env.spec.timestep_limit
                for i in range(0, num_rollout):
                    obs = env.reset()
                    steps = 0
                    done = False
                    while not done:
                        action = policy_fn(obs[None,:])     # action from policy
                        observations.append(obs)
                        actions.append(action)
                        obs, _, done, _ = env.step(action)  # take action in environment
                        steps += 1
                        if render:
                            env.render()
                        if steps % 500 == 0:
                            print("%i/%i" % (steps, max_steps))
                        if steps >= max_steps:
                            done = True
            else:
                # run expert policy on observations experienced by novice
                print('generating more training data for dataset aggregation')
                print('initial # of training samples:', expert_data['observations'].shape[0])

                for i in range(len(training_obs)):
                    env.reset()
                    obs = training_obs[i]
                    obs = np.reshape(obs, (1, self.obs_dim))
                    action = policy_fn(obs)
                    actions.append(action)
                    obs, _, _, _, = env.step(action)
                observations = training_obs
        
        observations = np.array(observations)
        actions = np.array(actions).reshape(len(actions), self.act_dim)

        # save expert data to file
        if dagger:
            expert_data = {'observations': np.concatenate((expert_data['observations'], observations), axis=0),
                            'actions': np.concatenate((expert_data['actions'], actions), axis=0)}
            print('final # of training samples:', expert_data['observations'].shape[0])
        else:
            expert_data = {'observations': observations,
                            'actions': actions}
        
        with open(self.expert_data_dir + expert_data_fname, 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

        return expert_data_fname


    def train_behavior_clone(self,
            training_data_fname,
            restore,
            model_fname=None,
            iter_train=1000,
            plot_loss=True,
            save_model=True):
        """ """
        training_data = self.load_expert_data(training_data_fname)
        if restore:
            if model_fname is None:
                print('must provide model filename for restore in behavior clone')
                raise Exception
            else:
                self.restore_model(model_fname)
        else:
            self.initialize()

        losses = []
        print('training on: ' + str(training_data['observations'].shape[0]) + ' samples.')
        for i in range(0, iter_train):
            feed_dict= {
                self.obs_train : training_data['observations'],
                self.act_train : training_data['actions']}
            _, loss = self.session.run([self.train_step, self.loss],
                feed_dict=feed_dict)
            losses.append(loss)
            if i % 100 == 0:
                print('iteration', i, '\t\tloss:', loss)

        if plot_loss:
            plt.plot(losses)
            plt.title('loss vs. iteration')
            plt.show()

        if save_model:
            if model_fname is None:
                model_fname = self.envname
            self.save_model(model_fname)

        return losses

    def train_dagger(self, 
            expert_policy_file,
            restore,
            model_fname=None,
            expert_data_fname=None,
            num_epoch=3,
            num_expert_rollout=1,
            iter_train=1000,
            iter_test_novice=500,
            plot_loss=True,
            save_model=True):
        """ 
        TODO 
        """
        if expert_data_fname is None:
            print('generating initial expert policy data...')
            expert_data_fname = self.run_expert_policy(
                expert_policy_file, render=False, num_rollout=num_expert_rollout)
        
        reward_list = []
        loss_list = []
        for i in range(num_epoch):
            # always restore between training epochs, but don't try to restore a nonexistent model
            if i > 0:
                restore = True

            # 1) train novice policy on expert data
            print('\nepoch:', i+1)
            epoch_loss = self.train_behavior_clone(
                expert_data_fname, restore, model_fname=model_fname,
                iter_train=iter_train, plot_loss=plot_loss, save_model=save_model)
            print('final loss after %i iterations: %f' %(iter_train, epoch_loss[-1]))
            loss_list += epoch_loss

            # 2) run novice policy, save data
            print('testing novice policy')
            novice_data_fname, rew = self.run_novice_policy(
                model_fname=model_fname, render=False, num_iter=iter_test_novice, 
                num_rollout=1, restore=restore, plot=False)
            reward_list += rew

            # 3) generate new training data by feeding novice observations to expert policy
            print('running expert policy on novice')
            expert_data_fname = self.run_expert_policy(
                expert_policy_file, novice_data_fname=novice_data_fname, 
                expert_data_fname=expert_data_fname, dagger=True)

        return reward_list, loss_list

    def initialize(self):
        self.session.run(self.init_vars)
        return

    def save_model(self,fname):
        self.saver.save(self.session, self.novice_policy_dir + fname)
        print('model saved at:', self.novice_policy_dir + fname)
        return
    
    def restore_model(self, fname):
        print('restorimg model saved at:', self.novice_policy_dir + fname)
        self.saver.restore(self.session, self.novice_policy_dir + fname)
        return

    def load_novice_data(self, novice_data_fname):
        try:
            with open(self.novice_data_dir + novice_data_fname, 'rb') as handle:
                novice_data = pickle.load(handle)
        except:
            print('unable to load novice data with given filename', novice_data_fname)
        return novice_data

    def load_expert_data(self, expert_data_fname):
        try:
            with open(self.expert_data_dir + expert_data_fname, 'rb') as handle:
                expert_data = pickle.load(handle)
        except:
            print('unable to load expert data with given filename', expert_data_fname)
        return expert_data








