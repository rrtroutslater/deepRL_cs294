import os 

import numpy as np 
import matplotlib.pyplot as plt 

from model_based_policy_rr import ModelBasedPolicy 
import utils
from logger import logger
from timer import timeit

class ModelBasedRL(object):

    def __init__(self,
            env,
            num_init_random_rollouts=10,
            max_rollout_length=500,
            num_onpolicy_iters=10,
            num_onpolicy_rollouts=10,
            training_epochs=60,
            training_batch_size=512,
            render=False,
            mpc_horizon=15,
            num_random_action_selection=4096,
            nn_layers=2):

        self.env = env
        self.num_init_random_rollouts = num_init_random_rollouts
        self.max_rollout_length = max_rollout_length
        self.num_onpolicy_iters = num_onpolicy_iters
        self.num_onpolicy_rollouts = num_onpolicy_rollouts
        self.training_epochs = training_epochs
        self.training_batch_size = training_batch_size
        self.render = render

        logger.info('Gathering random dataset')
        self.random_dataset = self.gather_rollouts(utils.RandomPolicy(env),
                                num_init_random_rollouts)

        logger.info('Creating policy')
        self.policy = ModelBasedPolicy(env, 
                        self.random_dataset,
                        horizon=mpc_horizon,
                        num_random_action_selection=num_random_action_selection)

        timeit.reset()
        timeit.start('total')

    def gather_rollouts(self, policy, num_rollouts):

        dataset = utils.Dataset()

        for _ in range(num_rollouts):
            state = self.env.reset()
            done = False 
            t = 0
            while not done:
                if self.render:
                    timeit.start('render')
                    self.env.render()
                    timeit.stop('render')
                timeit.start('get action')
                action = policy.get_action(state)
                timeit.stop('get action')
                timeit.start('env step')
                next_state, reward, done, _ = self.env.step(action)
                timeit.stop('env step')
                done = done or (t >= self.max_rollout_length)
                dataset.add(state, action, next_state, reward, done)

                state = next_state
                t += 1
        return dataset

    def train_policy(self, dataset):
        """
        trains the model-based policy
        """
        timeit.start('train policy')

        losses = []
        for _ in range(self.training_epochs):
            loss_total = 0.0
            num_data = 0

            d = dataset.random_iterator(self.training_batch_size)
            for states, actions, next_states, _, _ in d:
                loss = self.policy.train_step(states, actions, next_states)
                loss_total += loss
                num_data += 1

            losses.append(loss / num_data)
        # plt.plot(losses)
        # plt.show()
        logger.record_tabular('TrainingLossStart', losses[0])
        logger.record_tabular('TrainingLossFinal', losses[-1])

        timeit.stop('train policy')
        return

    def log(self, dataset):
        timeit.stop('total')
        dataset.log()
        logger.dump_tabular(print_func=logger.info)
        logger.debug('')
        for line in str(timeit).split('\n'):
            logger.debug(line)
        timeit.reset()
        timeit.start('total')

    def run_q1(self):
        """
        train on a dataset, see how good learned dynamics model predictions are
        """
        logger.info('Training policy....')
        self.train_policy(self.random_dataset)

        logger.info('Evaluating predictions...')
        for r_num, (states, actions, _, _, _) in enumerate(self.random_dataset.rollout_iterator()):
            pred_states = []

            pred = states[0]
            pred_states.append(pred)
            for i in range(0, states.shape[0]):
                pred = self.policy.predict(pred, actions[i])
                pred_states.append(pred)

            states = np.asarray(states)
            pred_states = np.asarray(pred_states)

            state_dim = states.shape[1]
            rows = int(np.sqrt(state_dim))
            cols = state_dim // rows
            f, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
            f.suptitle('Model predictions (red) versus ground truth (black) for open-loop predictions')
            for i, (ax, state_i, pred_state_i) in enumerate(zip(axes.ravel(), states.T, pred_states.T)):
                ax.set_title('state {0}'.format(i))
                ax.plot(state_i, color='k')
                ax.plot(pred_state_i, color='r')
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)
            f.savefig(os.path.join(logger.dir, 'prediction_{0:03d}.jpg'.format(r_num)), bbox_inches='tight')

        logger.info('All plots saved to folder')
        return

    def run_q2(self):
        """
        train model-based policy on random dataset, evaluate performance of the resulting policy
        """
        logger.info('Random policy')
        self.log(self.random_dataset)

        logger.info('Training policy ... ')
        self.train_policy(self.random_dataset)

        logger.info('Evaluating Policy')
        eval_dataset = self.gather_rollouts(self.policy, self.num_init_random_rollouts)

        logger.info('Trained policy')
        self.log(eval_dataset)
        return

    def run_q3(self):
        """
        start with random dataset, train policy on dataset, gather rollouts with policy, add to 
        dataset, repeat
        """
        dataset = self.random_dataset

        itr = -1
        logger.info('Iteration {0}'.format(itr))
        logger.record_tabular('Itr', itr)
        self.log(dataset)

        for itr in range(self.num_onpolicy_iters + 1):
            logger.info('Iteration {0}'.format(itr))
            logger.record_tabular('Itr', itr)

            logger.info('Training policy...')
            self.train_policy(self.random_dataset)

            logger.info('Gathering rollouts...')
            new_dataset = self.gather_rollouts(self.policy, self.num_onpolicy_rollouts)

            logger.info('Appending dataset...')
            dataset.append(new_dataset)

            self.log(new_dataset)
