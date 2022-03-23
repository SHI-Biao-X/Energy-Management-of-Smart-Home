# -*- coding: utf-8 -*-
"""
@author: Biao
"""
import argparse
import time
import matplotlib.pyplot as plt
import os

import numpy as np
import tensorflow as tf

import tensorlayer as tl

import DefProblem

tl.logging.set_verbosity(tl.logging.DEBUG)

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
print(parser.parse_args())
args = parser.parse_args()

#####################  hyper parameters  ####################

ALG_NAME = 'AC'
TRAIN_EPISODES = 100  # number of overall episodes for training
TEST_EPISODES = 10  # number of overall episodes for testing
MAX_STEPS = 24  # maximum time step in one episode
LAM = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

N_F = 4 # dimension of states
N_A = 11 # number of actions



###############################  Actor-Critic  ####################################


class Actor(object):

    def __init__(self, state_dim, action_num, lr=0.001):

        input_layer = tl.layers.Input([None, state_dim], name='state')
        layer = tl.layers.Dense(
            n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden1'
        )(input_layer)
        layer = tl.layers.Dense(
            n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden2'
        )(layer)
        layer = tl.layers.Dense(n_units=action_num, name='actions')(layer)
        self.model = tl.models.Model(inputs=input_layer, outputs=layer, name="Actor")

        self.model.train()
        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, state, action, td_error):
        with tf.GradientTape() as tape:
            _logits = self.model(np.array([state]))
            ## cross-entropy loss weighted by td-error (advantage),
            # the cross-entropy mearsures the difference of two probability distributions: the predicted logits and sampled action distribution,
            # then weighted by the td-error: small difference of real and predict actions for large td-error (advantage); and vice versa.
            _exp_v = tl.rein.cross_entropy_reward_loss(logits=_logits, actions=[action], rewards=td_error[0])
        grad = tape.gradient(_exp_v, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        return _exp_v

    def get_action(self, state, greedy=False):
        _logits = self.model(np.array([state]))
        _probs = tf.nn.softmax(_logits).numpy()
        if greedy:
            return np.argmax(_probs.ravel())
        return tl.rein.choice_action_by_probs(_probs.ravel())  # sample according to probability distribution

    def save(self):  # save trained weights
        path = os.path.join('model', '_'.join([ALG_NAME]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz(self.model.trainable_weights, name=os.path.join(path, 'model_actor.npz'))

    def load(self):  # load trained weights
        path = os.path.join('model', '_'.join([ALG_NAME]))
        tl.files.load_and_assign_npz(name=os.path.join(path, 'model_actor.npz'), network=self.model)


class Critic(object):

    def __init__(self, state_dim, lr=0.01):
        input_layer = tl.layers.Input([1, state_dim], name='state')
        layer = tl.layers.Dense(
            n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden1'
        )(input_layer)
        layer = tl.layers.Dense(
            n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden2'
        )(layer)
        layer = tl.layers.Dense(n_units=1, act=None, name='value')(layer)
        self.model = tl.models.Model(inputs=input_layer, outputs=layer, name="Critic")
        self.model.train()

        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, state, reward, state_):
        v_ = self.model(np.array([state_]))
        with tf.GradientTape() as tape:
            v = self.model(np.array([state]))
            ## TD_error = r + d * lambda * V(newS) - V(S)
            td_error = reward + LAM * v_ - v
            loss = tf.square(td_error)
        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        return td_error

    def save(self):  # save trained weights
        path = os.path.join('model', '_'.join([ALG_NAME]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz(self.model.trainable_weights, name=os.path.join(path, 'model_critic.npz'))

    def load(self):  # load trained weights
        path = os.path.join('model', '_'.join([ALG_NAME]))
        tl.files.load_and_assign_npz(name=os.path.join(path, 'model_critic.npz'), network=self.model)


if __name__ == '__main__':
    
    print("observation dimension: %d" % N_F)  # 4
    print("num of actions: %d" % N_A)  # 2 : left or right

    actor = Actor(state_dim=N_F, action_num=N_A, lr=LR_A)
    # we need a good teacher, so the teacher should learn faster than the actor
    critic = Critic(state_dim=N_F, lr=LR_C)

    t0 = time.time()
    if args.train:
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            state = DefProblem.init_s().astype(np.float32)
            step = 0  # number of step in this episode
            episode_reward = 0  # rewards of all steps
            
            if episode == TRAIN_EPISODES-1 :
                state_T_out = [state[2]]
                state_T_in = [state[3]]
                action_AC = []
                
            while True:
                step += 1
                
                action = actor.get_action(state)

                state_new, reward = DefProblem.step_s(action,state)
                state_new = state_new.astype(np.float32)

                episode_reward += reward

                try:
                    td_error = critic.learn(
                        state, reward, state_new)
                    # learn Value-function : gradient = grad[r + lambda * V(s_new) - V(s)]
                    actor.learn(state, action, td_error)  # learn Policy : true_gradient = grad[logPi(s, a) * td_error]
                except KeyboardInterrupt:  # if Ctrl+C at running actor.learn(), then save model, or exit if not at actor.learn()
                    actor.save()
                    critic.save()

                state = state_new
                
                if episode == TRAIN_EPISODES-1 :
                    state_T_out.append(state[2])
                    state_T_in.append(state[3])
                    action_AC.append(action*100)
                
                if step == MAX_STEPS - 1:
                    action = actor.get_action(state)
                    reward = DefProblem.reward(action, state)
                    episode_reward += reward
                    if episode == TRAIN_EPISODES-1 :
                        action_AC.append(action*100)
                    break

            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)

            print('Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}' \
                  .format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0))

        actor.save()
        critic.save()

        fig1 = plt.figure(1)
        plt.plot(all_episode_reward)
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME])))
        
        fig2 = plt.figure(2)
        
        ax1 = fig2.add_subplot(111)
        ax1.plot(state_T_in, label = 'Indoor Temperature')
        ax1.plot(state_T_out, label = 'Outdoor Temperature')
        ax1.set_ylabel('Temperature(℃)')
        ax1.set_xlabel('Time(hour)')
        
        ax2 = ax1.twinx()
        ax2.bar(range(24),action_AC,width=0.4, label = 'Energy Consumption of Air Conditioner')
        ax2.set_ylabel('Energy Consumption(Wh)')
        
        fig2.legend()
        fig2.savefig(os.path.join('image', '_'.join([ALG_NAME,'result'])))
        

    if args.test:
        actor.load()
        critic.load()

        for episode in range(TEST_EPISODES):
            episode_time = time.time()
            state = DefProblem.init_s().astype(np.float32)
            step = 0  # number of step in this episode
            episode_reward = 0
            while True:
                action = actor.get_action(state, greedy=True)
                state_new, reward = DefProblem.step_s(action,state)
                state_new = state_new.astype(np.float32)

                episode_reward += reward
                state = state_new
                step += 1

                if step == MAX_STEPS - 1:
                    action = actor.get_action(state)
                    reward = DefProblem.reward(action, state)
                    episode_reward += reward
                    print('Testing  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}' \
                          .format(episode + 1, TEST_EPISODES, episode_reward, time.time() - t0))
                    break