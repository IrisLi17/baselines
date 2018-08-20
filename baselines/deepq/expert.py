import pickle
import tensorflow as tf
import numpy as np
from baselines.deepq.memory import Memory
# from baselines.ddpg.ddpg import normalize, denormalize
import os
import math
import cv2
# import baselines.deepq.agc as agc
from baselines.deepq.agc import dataset,util

class Expert:
    def __init__(self, env):
        #self.limit = limit
        self.env = env
        #self.memory = Memory(limit=self.limit,
        #                     action_shape=self.env.action_space.shape,
        #                     #observation_shape=self.env.observation_space.shape
        #                     observation_shape=(84,84,4) )
        self.file_dir = None
        self.selected_traj = []

    def load_file(self, DATADIR, g):
        self.dataset = dataset.AtariDataset(DATADIR)
        self.DATADIR = DATADIR
        self.g = g
        stats = self.dataset.stats
        trajectories = self.dataset.trajectories


        # for g in stats.keys():
        top50_score = stats[g]['top50_score']
        for t in trajectories[g].keys():
            if trajectories[g][t][-1]['score'] > top50_score:
                self.selected_traj.append(t)
                #for frame in range(math.floor(len(trajectories[g][t])/4)-1):
                #    obs_path = os.path.join(SCREENDIR, str(g), str(t))
                #    obs0 = np.zeros((84,84,4))
                #    obs1 = np.zeros((84,84,4))
                #    for index in range(4):
                #        state0 = cv2.imread(os.path.join(obs_path,str(4*frame+index)+".png"))
                #        obs0[:,:,index] = agc.util.preprocess(state0)
                #        state1 = cv2.imread(os.path.join(obs_path,str(4*frame+4+index)+".png"))
                #        obs1[:,:,index] = agc.util.preprocess(state1)
                #    action = trajectories[g][t][4*frame]['action']
                #    reward = trajectories[g][t][4*frame]['reward']
                #    terminal = trajectories[g][t][4*frame]['terminal']
                #    self.memory.append(obs0,action,reward,obs1,terminal)



    def load_file_trpo(self, file_dir):
        self.file_dir = file_dir
        traj_data = np.load(file_dir)
        if self.limit is None:
            obs = traj_data["obs"][:]
            acs = traj_data["acs"][:]
        else:
            obs = traj_data["obs"][:self.limit]
            acs = traj_data["acs"][:self.limit]
        episode_num = len(acs)
        '''
        step_num = 0
        for i in range(episode_num):
            step_num += len(acs[i])
        print("Total Step is:", step_num, "\nTotal_Episode is:", episode_num)
        '''
        for i in range(episode_num):
            episode_len = len(acs[i])
            for j in range(episode_len):
                done = True if (j == episode_len - 1) else False
                self.memory.append(obs[i][j], acs[i][j], 0., 0., done)

    def sample(self, batch_size):
        traj = self.selected_traj[np.random.random_integers(len(self.selected_traj), size = 1)]
        nb_entries = len(self.dataset.trajectories[self.g][traj]) - 4
        frames = np.random.random_integers(nb_entries, size = batch_size)
        obs0_batch = [self.getFrameStack(traj, frame) for frame in frames]
        obs1_batch = [self.getFrameStack(traj, frame + 1) for frame in frames]
        reward_batch = [self.getReward(traj, frame) for frame in frames]
        action_batch = [self.dataset.trajectories[self.g][traj][frame]['action'] for frame in frames]
        terminal1_batch = [self.dataset.trajectories[self.g][traj][frame + 5]['terminal'] for frame in frames]
        result = {
            'obs0': obs0_batch,
            'obs1': obs1_batch,
            'rewards': reward_batch,
            'actions': action_batch,
            'terminals1': terminal1_batch,
        }
        return result

    def getFrameStack(self, traj, start, stack = 4):
        obs = np.zeros((84,84,stack))
        obs1 = np.zeros((84,84,stack))
        reward = 0.0
        obs_path = os.path.join(os.path.join(self.DATADIR, "screens"), str(self.g), str(traj))
        for frame in range(stack):
            state = cv2.imread(os.path.join(obs_path, str(start+frame) + ".png"))
            state1 = cv2.imread(os.path.join(obs_path, str(start+frame+1) + ".png"))
            obs[:,:,frame] = util.preprocess(state)
            obs1[:,:,frame] = util.preprocess(state1)
            reward += self.dataset.trajectories[self.g][traj][start+frame]['reward']
        action = self.dataset.trajectories[self.g][traj][start]['action']
        terminal = self.dataset.trajectories[self.g][traj][start]['terminal']
        return obs

    def getReward(self, traj, start, stack = 4):
        reward = 0.0
        for frame in range(stack):
            reward += self.dataset.trajectories[self.g][traj][start+frame]['reward']
        return reward
    # need to modify
#def set_tf(self, actor, critic, obs_rms, ret_rms, observation_range, return_range, supervise=False):
#    self.expert_state = tf.placeholder(tf.float32, shape=(None,) + self.env.observation_space.shape,
#                                       name='expert_state')
#    self.expert_action = tf.placeholder(tf.float32, shape=(None,) + self.env.action_space.shape,
#                                        name='expert_action')
#    normalized_state = tf.clip_by_value(normalize(self.expert_state, obs_rms),
#                                        observation_range[0], observation_range[1])
#    expert_actor = actor(normalized_state, reuse=True)
#    normalized_q_with_expert_data = critic(normalized_state, self.expert_action, reuse=True)
#    normalized_q_with_expert_actor = critic(normalized_state, expert_actor, reuse=True)
#    self.Q_with_expert_data = denormalize(
#        tf.clip_by_value(normalized_q_with_expert_data, return_range[0], return_range[1]), ret_rms)
#    self.Q_with_expert_actor = denormalize(
#        tf.clip_by_value(normalized_q_with_expert_actor, return_range[0], return_range[1]), ret_rms)
#    if supervise:
#        self.actor_loss = tf.nn.l2_loss(self.expert_action-expert_actor)
#        self.critic_loss = 0
#    else:
#        self.critic_loss = tf.reduce_mean(tf.nn.relu(self.Q_with_expert_actor - self.Q_with_expert_data))
#        self.actor_loss = -tf.reduce_mean(self.Q_with_expert_actor)
#    self.dist = tf.reduce_mean(self.Q_with_expert_data - self.Q_with_expert_actor)
#
