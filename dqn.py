import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append("./game/")
import wrapped_flappy_bird as bird_game
import cv2
from collections import deque
import random

def state_input(s, trainable, reuse):
	with tf.variable_scope('state', reuse = reuse):
		conv1 = tf.layers.conv2d(inputs = s, 
			filters = 32, 
			kernel_size = [10, 10], 
			strides = 4, 
			activation = tf.nn.relu, 
			use_bias = True,
			kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
			bias_initializer = tf.constant_initializer(0.1),
			kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
			bias_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
			trainable = trainable,
			reuse = reuse,
			name = 'conv1')
		pool1 = tf.layers.max_pooling2d(inputs = conv1,	pool_size = [2,2], strides = [2,2], name = 'pool1')
		conv2 = tf.layers.conv2d(inputs = pool1, 
			filters = 64, 
			kernel_size = [5, 5], 
			strides = 2, 
			activation = tf.nn.relu, 
			use_bias = True,
			kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
			bias_initializer = tf.constant_initializer(0.1),
			kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
			bias_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
			trainable = trainable,
			reuse = reuse,
			name = 'conv2')
		# pool2 = tf.layers.max_pooling2d(inputs = conv2,	pool_size = [2,2], strides = [2,2], name = 'pool2')
		conv3 = tf.layers.conv2d(inputs = conv2, 
			filters = 64, 
			kernel_size = [2, 2], 
			strides = 1, 
			activation = tf.nn.relu,
			use_bias = True,
			kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
			bias_initializer = tf.constant_initializer(0.1),
			kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
			bias_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
			trainable = trainable,
			reuse = reuse,
			name = 'conv3')
		# pool3 = tf.layers.max_pooling2d(inputs = conv3,	pool_size = [2,2], strides = [2,2], name = 'pool2')
		flat = tf.reshape(conv3, [-1, 256])
		dense1 = tf.layers.dense(inputs = flat, 
				units = 256, 
				activation = tf.nn.relu,
				use_bias = True,
				kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
				bias_initializer = tf.constant_initializer(0.1),
				kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
				bias_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
				trainable = trainable,
				name = 'dense1')
		return dense1

class Critic(object):
	def __init__(self, sess, action_dim, learning_rate):
		self.sess = sess
		self.a_dim = action_dim
		self.lr = learning_rate

		self.A = tf.placeholder(tf.float32, [None, self.a_dim])
		self.S = tf.placeholder(tf.float32, [None, 80, 80, 4])
		self.S_ = tf.placeholder(tf.float32, [None, 80, 80, 4])
		self.Y = tf.placeholder(tf.float32, [None])

		self.Scnn = state_input(self.S, trainable = True, reuse = False)
		self.Scnn_ = state_input(self.S_, trainable = False, reuse = True)

		with tf.variable_scope('Critic'):
			self.c = self.build_net(self.Scnn, scope = 'eval_net', trainable = True)
			self.c_ = self.build_net(self.Scnn_, scope = 'target_net', trainable = False)

			self.est = tf.reduce_sum(tf.multiply(self.c, self.A), axis = 1)
			self.cost = tf.reduce_mean(tf.square(self.Y - self.est))
		
			with tf.variable_scope('train'):
				self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

		self.params_e = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic/eval_net')
		self.params_t = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic/target_net')

		self.replace = [tf.assign(t, e) for t, e in zip(self.params_t, self.params_e)]
		self.replace_cnter = 0

	def build_net(self, s, scope, trainable):
		with tf.variable_scope(scope):
			denseS = tf.layers.dense(inputs = s, 
				units = 256, 
				use_bias = True,
				kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
				bias_initializer = tf.constant_initializer(0.1),
				kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
				bias_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
				trainable = trainable,
				name = 'denseS')
			# denseA = tf.layers.dense(inputs = a, 
			# 	units = 128, 
			# 	kernel_initializer = tf.random_normal(stddev = 0.01),
			# 	bias_initializer = tf.constant(0.01),
			# 	kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
			# 	bias_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
			# 	trainable = trainable,
			# 	name = 'denseA')
			merge = tf.nn.relu(denseS)# + denseA
			# dense2 = tf.layers.dense(inputs = merge, 
			# 	units = 32, 
			# 	use_bias = True,
			# 	kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
			# 	bias_initializer = tf.constant_initializer(0.1),
			# 	kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
			# 	bias_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
			# 	trainable = trainable,
			# 	name = 'dense2')
			value = tf.layers.dense(inputs = merge, 
				units = 1, 
				use_bias = True,
				kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
				bias_initializer = tf.constant_initializer(0.1),
				kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
				bias_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
				trainable = trainable,
				name = 'value')
			adv = tf.layers.dense(inputs = merge, 
				units = self.a_dim, 
				use_bias = True,
				kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
				bias_initializer = tf.constant_initializer(0.1),
				kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
				bias_regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7),
				trainable = trainable,
				name = 'adv')
			Qout = value + (adv - tf.reduce_mean(adv, axis = 1, keepdims = True))
		return Qout

	def learn(self, s, a, y):
		cst, op = self.sess.run([self.cost, self.train_op], feed_dict = {self.S : s, self.A : a, self.Y : y})
		return cst

	def choose_action(self, s):
		action = self.sess.run(self.c, feed_dict = {self.S : s})
		act = []
		for i in range(len(action)):
			act.append([1, 0] if action[i][0] >= action[i][1] else [0, 1])
		return act

C_LR = 0.000001
REPLAY_MEMORY = 50000
OBSERVE = 20000
EXPLORE = 2000000
BATCH = 32
GAMMA = 0.99
INITIAL_EPSILON = 0.0001
FINAL_EPSILON = 0.0001

def playGame():
	sess = tf.Session()
	critic = Critic(sess, action_dim = 2, learning_rate = C_LR)
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())

	#restore
	checkpnt = tf.train.get_checkpoint_state('./model/2560000/')
	saver.restore(sess, checkpnt.model_checkpoint_path)

	que = deque()

	game = bird_game.GameState()
	s, r, terminal = game.frame_step([1,0]) #do nothing
	s, r, terminal = game.frame_step([1,0])
	s = cv2.cvtColor(cv2.resize(s, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, s = cv2.threshold(s, 1, 255, cv2.THRESH_BINARY)
	s = np.reshape(s, (80, 80, 1))
	print(s)
	s = np.concatenate((s, s, s, s), axis = 2)
	s = np.reshape(s, (1, 80, 80, 4))

	t = 170000
	cost = 0
	sample = 0
	max_score = 0
	score_times = 0
	score = 0
	epsilon = INITIAL_EPSILON
	while 'flappy' != 'angry':

		if t > OBSERVE and epsilon > FINAL_EPSILON:
			epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

		if np.random.uniform() < epsilon:
			a = np.random.randint(0, 2)
			a = [1, 0] if a == 0 else [0, 1]
		else:
			a = critic.choose_action(s)[0]   #[1, 0] or [0, 1]

		s_, r, terminal = game.frame_step(a)   #r: (1)
		s_ = cv2.cvtColor(cv2.resize(s_, (80, 80)), cv2.COLOR_BGR2GRAY)
		ret, s_ = cv2.threshold(s_, 1, 255, cv2.THRESH_BINARY)
		s_ = np.reshape(s_, (1, 80, 80, 1))
		s_ = np.append(s_, s[:, :, :, :3], axis = 3)    #(80, 80, 4)

		if r == 1:
			score += 1
			score_times += 1
		if terminal:
			if max_score < score:
				max_score = score
			score = 0


		que.append((np.reshape(s, (80, 80, 4)), a, r, np.reshape(s_, (80, 80, 4)), terminal))
		if len(que) > REPLAY_MEMORY:
			que.popleft()

		if sample > OBSERVE:
			batch = random.sample(que, BATCH)

			sb = [b[0] for b in batch]
			ab = [b[1] for b in batch]
			rb = [b[2] for b in batch]
			s_b = [b[3] for b in batch]
			tmnl = [b[4] for b in batch]

			s_b = np.reshape(s_b, (BATCH, 80, 80, 4))
			a_b = critic.choose_action(s_b)
			val = sess.run(critic.c_, feed_dict = {critic.S_ : s_b})
			yb = []
			for i in range(BATCH):
				if tmnl[i]:
					yb.append(rb[i])
				else:
					yb.append(rb[i] + GAMMA * np.sum(a_b[i] * val[i]))
			cost = critic.learn(sb, ab, yb)
			
			critic.replace_cnter +=1
			if critic.replace_cnter % 1000 == 0:
				sess.run(critic.replace)

		s = s_
		t += 1
		sample += 1


		if t % 10000 == 0 and t != 190000 and t != 180000:
			# savePath = '.\\model\\' + str(t)
			savePath = './model/' + str(t)
			mdlName = 'model' + str(t)
			saver.save(sess, os.path.join(savePath, mdlName))
			print(max_score, ' ', score_times)

		print("TIMESTEP:", t, " ACTION:", np.argmax(a), " REWARD:", r, " cost:", '{:.5f}'.format(cost), ' eps:', epsilon)

playGame()