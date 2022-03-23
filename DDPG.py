# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:50:36 2022

@author: Biao
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl

import DefProblem

tl.logging.set_verbosity(tl.logging.DEBUG)

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

#####################  hyper parameters  ####################

ALG_NAME = 'DDPG'
TRAIN_EPISODES = 200  # total number of episodes for training
TEST_EPISODES = 10  # total number of episodes for training
MAX_STEPS = 24  # total number of steps for each episode

LR_A = 0.01  # learning rate for actor
LR_C = 0.02  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000  # size of replay buffer
BATCH_SIZE = 32  # update action batch size
VAR = 2  # control exploration

###############################  DDPG  ####################################


class DDPG(object):
    """
    DDPG class
    """

    def __init__(self, action_dim, state_dim, action_range):
        self.memory = np.zeros((MEMORY_CAPACITY, state_dim * 2 + action_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.action_dim, self.state_dim, self.action_range = action_dim, state_dim, action_range
        self.var = VAR

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            input_layer = tl.layers.Input(input_state_shape, name='A_input')
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(input_layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l2')(layer)
            layer = tl.layers.Dense(n_units=action_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(layer)
            layer = tl.layers.Lambda(lambda x: action_range * x)(layer)
            return tl.models.Model(inputs=input_layer, outputs=layer, name='Actor' + name)

        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            state_input = tl.layers.Input(input_state_shape, name='C_s_input')
            action_input = tl.layers.Input(input_action_shape, name='C_a_input')
            layer = tl.layers.Concat(1)([state_input, action_input])
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l2')(layer)
            layer = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(layer)
            return tl.models.Model(inputs=[state_input, action_input], outputs=layer, name='Critic' + name)

        self.actor = get_actor([None, state_dim])
        self.critic = get_critic([None, state_dim], [None, action_dim])
        self.actor.train()
        self.critic.train()

        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.actor_target = get_actor([None, state_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        self.critic_target = get_critic([None, state_dim], [None, action_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

    def ema_update(self):
        """
        Soft updating by exponential smoothing
        :return: None
        """
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def get_action(self, s, greedy=False):
        """
        Choose action
        :param s: state
        :param greedy: get action greedy or not
        :return: act
        """
        a = self.actor(np.array([s], dtype=np.float32))[0]
        if greedy:
            return a
        return np.clip(
            np.random.normal(a, self.var), -self.action_range, self.action_range
        )  # add randomness to action selection for exploration

    def learn(self):
        """
        Update parameters
        :return: None
        """
        self.var *= .9995
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        datas = self.memory[indices, :]
        states = datas[:, :self.state_dim]
        actions = datas[:, self.state_dim:self.state_dim + self.action_dim]
        rewards = datas[:, -self.state_dim - 1:-self.state_dim]
        states_ = datas[:, -self.state_dim:]

        with tf.GradientTape() as tape:
            actions_ = self.actor_target(states_)
            q_ = self.critic_target([states_, actions_])
            y = rewards + GAMMA * q_
            q = self.critic([states, actions])
            td_error = tf.losses.mean_squared_error(y, q)
        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            a = self.actor(states)
            q = self.critic([states, a])
            actor_loss = -tf.reduce_mean(q)  # maximize the q
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        self.ema_update()

    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def save(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic_target.hdf5'), self.critic_target)

    def load(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic_target.hdf5'), self.critic_target)


if __name__ == '__main__':

    state_dim = 4
    action_dim = 1
    action_range = 500  # scale action, [-action_range, action_range]

    agent = DDPG(action_dim, state_dim, action_range)

    t0 = time.time()
    if args.train:  # train
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            state = DefProblem.init_s()
            episode_reward = 0
            step = 0
            
            if episode == TRAIN_EPISODES-1 :
                state_T_out = [state[2]]
                state_T_in = [state[3]]
                action_AC = []
                
            while True:
                step += 1
                # Add exploration noise
                action = agent.get_action(state)
                state_, reward = DefProblem.step_s_DDPG(action[0],state)
                agent.store_transition(state, action, reward, state_)

                if agent.pointer > MEMORY_CAPACITY:
                    agent.learn()
                
                state = state_
                episode_reward += reward
                
                if episode == TRAIN_EPISODES-1 :
                    state_T_out.append(state[2])
                    state_T_in.append(state[3])
                    action_AC.append(action[0]+500)
                
                if step == MAX_STEPS - 1:
                    action = agent.get_action(state)
                    reward = DefProblem.reward_DDPG(action[0], state)
                    episode_reward += reward
                    if episode == TRAIN_EPISODES-1 :
                        action_AC.append(action[0]+500)
                    break
                
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            
            print('Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}' \
                  .format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0))
        
        agent.save()
        
        fig1 = plt.figure(1)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME])))
        
        fig2 = plt.figure(2)
        
        ax1 = fig2.add_subplot(111)
        ax1.plot(state_T_in, label = 'T_in')
        ax1.plot(state_T_out, label = 'T_out')
        
        ax2 = ax1.twinx()
        ax2.bar(range(24),action_AC,width=0.4)
        
        fig2.savefig(os.path.join('image', '_'.join([ALG_NAME,'result'])))

    # if args.test:
    #     # test
    #     agent.load()
    #     for episode in range(TEST_EPISODES):
    #         state = env.reset()
    #         episode_reward = 0
    #         for step in range(MAX_STEPS):
    #             env.render()
    #             state, reward, done, info = env.step(agent.get_action(state, greedy=True))
    #             episode_reward += reward
    #             if done:
    #                 break
    #         print(
    #             'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
    #                 episode + 1, TEST_EPISODES, episode_reward,
    #                 time.time() - t0
    #             )
    #         )