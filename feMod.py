import math
import numpy as np
import os
import gym
import random
from .utils.mygym import convert_to_gym
from keras.datasets import boston_housing

from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse

class featureModel(object):
	# Initialize simulation
    stepsize = 0.01

    model = None
    state = None
    state0 = None
  
    istep = 0

    state_desc_istep = None
    prev_state_desc = None
    state_desc = None

    initial_dataset = None
    prev_dataset =  None
    curr_dataset = None

    x_train = None
	y_train = None

	x_test =  None
	y_test = None

	action_space_size = 17

    def __init__(self):
    	(self.x_train, self.y_train), (self.x_test, self.y_test) = boston_housing.load_data()
    	self.initial_dataset = x_train
    	self.curr_dataset = x_train


    def get_engineered(self,action):
    	print(action)

    	cols = []
    	isadd = 0
    	ismul = 0
    	isdiv = 0
    	added_col = None
    	mult_col = None
    	div_col = None

    	isdrop = 0
    	for i in range(0,13):
    		if(action[i]>=0.8):
    			cols.append(i)
    	if(action[13]>=0.8):
    		isadd = 1
    	if(action[14]>=0.8):
    		ismul = 1
    	if(action[15]>=0.8):
    		isdiv = 1
    	if(action[16]>=0.8):
    		isdrop = 1

    	data = self.curr_dataset

    	if(len(cols)>1):

	    	if(isadd):
	    		for i in range(0,len(cols)):
	    			added_col += data[:,cols[i]]

	    	if(ismul):
	    		for i in range(0,len(cols)):
	    			mult_col += data[:,cols[i]]
	    	if(isdiv):
	    		# for i in range(0,len(cols)):
	    		# 	added_col += data[:,cols[i]]


	    if(added_col is not None):
	    	added_col=added_col.reshape(added_col.shape[0],1)
	    	data = np.concatenate((data,added_col),axis=1)

	    if(mult_col is not None):
	    	mult_col=mult_col.reshape(mult_col.shape[0],1)
	    	data = np.concatenate((data,mult_col),axis=1)

	    if(isdrop):
	    	data = np.delete(data,cols[0], 1)  

	    return data

	def actuate(self,action):
		if np.any(np.isnan(action)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")

        self.last_action = action

    	self.curr_dataset = get_engineered(action)


    	return self.curr_dataset

    # def get_state_desc(self):
    #     if self.state_desc_istep != self.istep:
    #         self.prev_state_desc = self.state_desc
    #         self.state_desc = self.compute_state_desc()
    #         self.state_desc_istep = self.istep
    #     return self.state_desc

    def reset(self):
        self.curr_dataset = initial_dataset
        # self.prev_dataset = initial_dataset	
        # self.state.setTime(0)
        self.istep = 0

    def integrate(self):
        # Define the new endtime of the simulation
        self.istep = self.istep + 1

    def get_action_space_size(self):
        return self.action_space_size






class feDL(feEnv):
	regr = None
	
	x_train = None
	y_train = None

	x_test =  None
	y_test = None

    def __init__(self):
    	self.regr = linear_model.LinearRegression()
    	(self.x_train, self.y_train), (self.x_test, self.y_test) = boston_housing.load_data()
        


    def is_done(self):
        
        return 1

    
    def get_observation(self):

        res = []
        

        return res


    def get_observation_space_size(self):
 
        return 100

    def get_action_space_size(self):

        return 17



    def reset(self):
        return super().reset()

    def reward(self,resulted_datast):
        # state_desc = self.get_state_desc()
        self.regr.fit(resulted_datast,self.y_train)
        pred = regr.predict(self.x_test)
        ms_error = mse(self.y_test,pred)
        return -ms_error

    def change_model(self):


class Spec(object):
    def __init__(self, *args, **kwargs):
        self.id = 0
        self.timestep_limit = 300

class feEnv(gym.Env):
    action_space = None
    observation_space = None
    istep = 0
    feature_model = None
    spec = None
    time_limit = 1e10

    prev_state_desc = None
    prev_dataset = None


    def reward(self):
        raise NotImplementedError

    def is_done(self):
        return False

    def __init__(self):
        self.integrator_accuracy = integrator_accuracy
        self.load_model()

    def load_model(self, model_path = None):

    	self.feature_model = featureModel()

        # Create specs, action and observation spaces mocks for compatibility with OpenAI gym
        self.spec = Spec()
        self.spec.timestep_limit = self.time_limit

		# self.observation_space = ( [-math.pi*100] * self.get_observation_space_size(), [math.pi*100] * self.get_observation_space_s
        self.observation_space = ( [0] * self.get_observation_space_size(), [0] * self.get_observation_space_size() )
    	self.action_space = ( [0.0] * self.feature_model.get_action_space_size(), [1.0] * self.feature_model.get_action_space_size() )

        self.action_space = convert_to_gym(self.action_space)
        self.observation_space = convert_to_gym(self.observation_space)

    



    def get_state_desc(self):
        return self.feature_model.get_state_desc()


    # def get_engineered_observation(self):
    # 	current_state_desc = self.get_state_desc()

    # 	return self.get_engineered(current_state_desc,)

    def get_prev_state_desc(self):
        return self.prev_state_desc

    # def get_observation(self):
    #     # This one will normally be overwrtitten by the environments
    #     # In particular, for the gym we want a vector and not a dictionary
    #     return self.osim_model.get_state_desc()

    def get_observation_space_size(self):
        return 0

    def get_action_space_size(self):
        return 0

    def reset(self):
        self.feature_model.reset()

        if not project:
            return self.get_state_desc()
        return self.get_observation()

    def step(self, action):
        # self.prev_state_desc = self.get_state_desc()
        # self.prev_dataset = self.get_dataset()
        resulted_dataset = self.feature_model.actuate(action)

        obs = run_model(resulted_dataset)

        return [ obs, self.reward(resulted_dataset), self.is_done() or (self.feature_model.istep >= self.spec.timestep_limit), {} ]

    def render(self, mode='human', close=False):
        return