import tensorflow as tf
import numpy as np
import utils


class ModelBasedPolicy():
    def __init__(self,
                env,
                init_dataset,
                horizon=15,
                num_random_action_selection=4096,
                nn_layers=1,
                learning_rate=1e-3):

        self.cost_fn = env.cost_fn
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_space_low = env.action_space.low
        self.action_space_high = env.action_space.high
        self.init_dataset = init_dataset
        self.horizon = horizon
        self.num_random_action_selection = num_random_action_selection
        self.nn_layers = nn_layers
        self.learning_rate = learning_rate

        self.sess, self.state_ph, self.action_ph, self.next_state_ph, \
            self.next_state_pred, self.loss, self.optimizer, self.best_action = self.setup_graph()

    def setup_placeholders(self):
        """
        creates placeholders for trainig, prediction, and action selection

        returns:
            state_ph: current state
            action_ph: current action 
            next_state_ph: next state
        """
        state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        action_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim])
        next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        return state_ph, action_ph, next_state_ph

    def dynamics_func(self, state, action, reuse):
        """
        takes state and action, returns the next state 

        returns:
            prediction of next state
        """
        state_norm = utils.normalize(
            state, self.init_dataset.state_mean, self.init_dataset.state_std)
        action_norm = utils.normalize(
            action, self.init_dataset.action_mean, self.init_dataset.action_std)

        # network input is concatenated state, action
        s_a = tf.concat([state_norm, action_norm], axis=1)
        d_next_state_norm = utils.build_mlp(
            s_a, self.state_dim, 'prediction', self.nn_layers, reuse=reuse)

        d_next_state = utils.unnormalize(
            d_next_state_norm, self.init_dataset.delta_state_mean, self.init_dataset.delta_state_std)

        next_state_pred = d_next_state + state
        return next_state_pred

    def setup_training(self, state_ph, next_state_ph, next_state_pred):
        """
        inputs:
            current state, next state, predicted next state

        returns:
            loss, and optimizer for training the dynamics model
        """
        d_state = next_state_ph - state_ph
        d_state_pred = next_state_pred - state_ph

        d_state_pred_norm = utils.normalize(
            d_state_pred, self.init_dataset.delta_state_mean, self.init_dataset.delta_state_std)

        d_state_norm = utils.normalize(
            d_state, self.init_dataset.delta_state_mean, self.init_dataset.delta_state_std)

        # loss = tf.losses.mean_squared_error(d_state_pred, next_state_pred)
        loss = tf.nn.l2_loss(d_state_pred_norm - d_state_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        return loss, optimizer

    def setup_action_selection(self, state_ph):
        """
        compute best action from current state by using randomly sampled action sequences to predict
        future states.  evaluate these predictions according to a cost function, select action
        sequence with the lowest cost. 

        inputs:
            current state

        returns:
            first action in sequence minimizing cost
        """
        total_costs = np.zeros(shape=(self.num_random_action_selection,))

        # iterate over each action in the horizon
        for i in range(0, self.horizon):
            # batch of actions at each timestep -> parallel simulation
            random_act_na = tf.random_uniform(
                shape=[self.num_random_action_selection, self.action_dim],
                minval=self.action_space_low,
                maxval=self.action_space_high )

            # store first action taken (used to evaluate best action given resulting trajectory)
            if i == 0:
                first_acts = random_act_na

            # propagate dynamics for each action, store resulting costs
            state_ph = state_ph 
            next_state_pred_ph = self.dynamics_func(state_ph, random_act_na, reuse=True)
            cost = self.cost_fn(state_ph, random_act_na, next_state_pred_ph)
            state_ph = next_state_pred_ph
            total_costs += cost 

        # best action = action which leads to lowest total cost trajectory
        best_trajectory_idx = tf.argmin(total_costs, axis=0)
        best_act = first_acts[best_trajectory_idx]
        return best_act

    def setup_graph(self):
        """
        set up tensorflow graph for training, prediction, and action selection
        """
        sess = tf.Session()

        state_ph, action_ph, next_state_ph = self.setup_placeholders()
        next_state_pred = self.dynamics_func(state_ph, action_ph, reuse=False)
        loss, optimizer = self.setup_training(state_ph, next_state_ph, next_state_pred)
        best_action = self.setup_action_selection(state_ph)
        sess.run(tf.global_variables_initializer())
        return sess, state_ph, action_ph, next_state_ph, next_state_pred, loss, optimizer, best_action

    def train_step(self, states, actions, next_states):
        """
        performs one step of gradient descent

        returns:
            loss from performing gradient descent 
        """
        loss, _ = self.sess.run(
            [self.loss, self.optimizer],
            feed_dict={
                self.state_ph : states,
                self.action_ph : actions,
                self.next_state_ph : next_states})
        return loss

    def predict(self, state, action):
        """
        predit next state given current state and action

        returns:
            predicted next state
        """
        assert np.shape(state) == (self.state_dim,)
        assert np.shape(action) == (self.action_dim,)
        
        next_state_pred = self.sess.run(
            self.next_state_pred,
            feed_dict={
                self.state_ph : [state],
                self.action_ph : [action]})
        next_state_pred = next_state_pred[0]

        assert np.shape(next_state_pred) == (self.state_dim,)
        return next_state_pred

    def get_action(self, state):
        """
        uses samples to estimate the action minimizing the cost function, given the current states

        returns:
            best action
        """
        assert np.shape(state) == (self.state_dim,)

        # duplicate the states to account for multiple actions tested for each simulation step
        states = np.tile(state, (self.num_random_action_selection,1))

        best_action = self.sess.run(
            self.best_action,
            feed_dict={
                self.state_ph : states })

        assert np.shape(best_action) == (self.action_dim,)
        return best_action




