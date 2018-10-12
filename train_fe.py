# Derived from keras-rl
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam
import keras.backend as K

import numpy as np

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from feMod import *



from keras.optimizers import RMSprop

import argparse
import math

# we will read the data and do some preprocessing to get it ready for RL

from keras.datasets import boston_housing


parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=100000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--token', dest='token', action='store', required=False)
args = parser.parse_args()

env = feDL()
observation_space_shape = env.observation_space.shape
nb_actions = env.action_space.shape[0]

actor = Sequential()
actor.add(Flatten(input_shape=(1,) + observation_space_shape))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + observation_space_shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = concatenate([action_input, flattened_observation])
x = Dense(20)(x)
x = Activation('relu')(x)
# x = Dense(64)(x)
# x = Activation('relu')(x)
# x = Dense(64)(x)
# x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


memory = SequentialMemory(limit=50000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.get_action_space_size())
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=320, nb_steps_warmup_actor=320,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.,batch_size=256)
# agent = ContinuousDQNAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
#                            memory=memory, nb_steps_warmup=1000, random_process=random_process,
#                            gamma=.99, target_model_update=0.1)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Training here 
# Loading weights for retrain. commet this if you are training from scratch - 
# agent.load_weights('/home/ajwahir/sads/cookie/models/batch200/interval_5')

# Okay, now it's time to learn something! We capture the interrupt exception so that training
# can be prematurely aborted. Notice that you can the built-in Keras callbacks!
weights_filename = 'ddpg_{}_weights'.format('big_head_1k')
checkpoint_weights_filename = 'ddpg_' + 'big_head_1k' + '_weights_{step}'
log_filename = 'ddpg_{}_log.json'.format('big_head_1k')
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=10000)]
callbacks += [FileLogger(log_filename, interval=100)]

agent.fit(env, nb_steps=nallsteps, visualize=False, verbose=1, nb_max_episode_steps=env.time_limit, log_interval=100,callbacks=callbacks)
# After training is done, we save the final weights.
agent.save_weights(args.model, overwrite=True)