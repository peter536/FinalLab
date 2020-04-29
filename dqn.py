import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from PIL import Image
import gym
import random

######
#
# Alex Peterson, Purdue University CS390 NIP Final Project
#
######
#=========================<VARIABLES>================================================
# The number of games that the network will play
NUM_EPISODES = 225

# The highest number of frames we allow per game before we move on
MAX_LENGTH_FOR_EPISODE = 10000

# How often we consider a frame
FREQUENCY = 4

# The size of the processed screen
IMG_SIZE = (84, 110)

# The number of steps before we update the target net
TARGET_NET_UPDATE_FREQ = 10000 

# The greatest number of states that we compare our q from
MAX_STATES_COMP = 25

# The smallest size we allow our Replay or Memory of Events to be
REPLAY_START_SIZE = 10000 

# The largest size we allow our Replay to be
REPLAY_MEM_SIZE = 50000

# Size of the minibatches used in training
MINIBATCH_SIZE = 32

# Discount factor for future points
GAMMA = 0.99

#Exploration Rate: Starts at 1
EPSILON = 1

#Rate at which it linearly decays 
EPSILON_DECAY =  0.00004

# The exploration rate cannot go lower!
EPSILON_MIN = 0.1

# Number of epochs for the DQN in training
EPOCHS = 1

# The Atari 2600 game in use
GAME = 'Assault-v0'
#GAME = 'Breakout-v0'

# Whether to load weights or not
WILL_LOAD_WEIGHTS = 0
#WILL_LOAD_WEIGHTS = 1


# Paths for the dqn and target dqn weights to load/save
DQN_WEIGHTS_PATH = "/Users/alex_p/Desktop/CS390NIP/FinalLab/dqn_weights.h5"
TARGET_WEIGHTS_PATH = "/Users/alex_p/Desktop/CS390NIP/FinalLab/target_weights.h5"


#=========================<DQN CODE>================================================

# Class for the Network for ease of use. 
class DeepQNetwork:
  # Initializes the DQN
  def __init__(self, actions, input_shape, minibatches, gamma, load_fp):
    self.actions = actions		# Network Output Size/Number of Available Actions
    self.input_shape = input_shape	# Input Shape for the DQN
    self.minibatches = minibatches	# Minibatch Size for the DQN
    self.gamma = gamma			# Discount rate, as described earlier
    self.load_fp = load_fp		# Load Filepath for weights
    
    # The actual model for the DQN
    self.model = self.build_model(self.actions, self.input_shape)
  ## END __INIT__ ##


  # Loss function I read that is used by DQNs
  # as read at: https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4
  def huber_loss(x, y):
    err = K.abs(x - y)
    quadratic = K.clip(err, 0.0, 1.0)
    linear = err - quadratic
    loss = K.mean(0.5 * K.square(quadratic) + linear, axis=-1)
    return loss


  # Constructs the DQN Model, using the architecture used from the Nature Article
  # found at: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf 
  def build_model(self, acts, shape):
    print("Building Deep Q Network\n")
    model = Sequential()
    model.add(Conv2D(32, 8, strides=(4,4), padding='valid', activation='relu', input_shape=shape, data_format='channels_first'))
    model.add(Conv2D(64, 4, strides=(2,2), padding='valid', activation='relu', input_shape=shape, data_format='channels_first'))
    model.add(Conv2D(64, 3, strides=(1,1), padding='valid', activation='relu', input_shape=shape, data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu')) # this is the rectifier mentioned in the paper
    model.add(Dense(self.actions))
    #### potentially load model ########
    if WILL_LOAD_WEIGHTS == 1:
      model.load_weights(load_fp)
    model.compile(loss=[DeepQNetwork.huber_loss], optimizer='rmsprop', metrics=['accuracy'])    
    return model
  ## END BUILD_MODEL ##

  # Provided a batch, generates inputs and targets for the neural net
  def train_model(self, batch, target_net):
    
    xTrain = []		# Inputs
    targetTrain = []	# Targets
    
    # For each element in the provided batch
    for item in batch:	
      xTrain.append(item['source'].astype(np.float64)) 	# Take the state from the item, add to inputs
      
      n_state = item['dest'].astype(np.float64)		# Next State. Is the next state from the item
      
      # Create the target, add the predictions from the nontarget DQN on the current item's current state
      # Take from the current state of the item, where all actions have 0 except the action taken
      target = list(self.predict_model(item['source'])[0])

      # Use the target network to predict for the next state.
      # Flatten the output to a list of the predictions. These will be Q values.
      n_pred = target_net.predict_model(n_state).ravel()
      # Select the next Q by choosing the maximum out of the predictions.
      next_q = np.max(n_pred)
      
      # if this is the final state
      if item['final']:
        # set index of action to be the reward
        target[item['action']] = item['reward']
      else:
        # otherwise, if not last one ('final' is zero)
        # Make the value at the index of action equal to the reward plus future reward times the discount
        target[item['action']] = item['reward'] + self.gamma * next_q
      
      # Add the target (a list of all actions with a 0 or a Q value
      targetTrain.append(target)
    ## END FOR ##
    # Change formet of the inputs and targets prior to fitting the non-target DQN
    xTrain = np.asarray(xTrain).squeeze()
    targetTrain = np.asarray(targetTrain).squeeze()
    trained_model = self.model.fit(xTrain, targetTrain, batch_size=MINIBATCH_SIZE, epochs=EPOCHS)
    return trained_model
  ## END TRAIN_MODEL ## 


  # Given a state, return Q-values for each action
  def predict_model(self, state):
    state = state.astype(np.float64)
    return self.model.predict(state, batch_size=1)
  ## END PREDICT_MODEL ##


#=========================<AGENT CODE>================================================

# Class for the Agent, the big kahuna that holds all the values
class QNetAgent:

  # Initializes the Agent with appropriate parameters, a DQN, and a Target DQN
  def __init__(self, actions, net_input_shape, replay_mem_size, minibatches, gamma, epsilon, epsilon_decay, epsilon_min):
    self.actions = actions			# Number of actions, Network outputs
    self.net_input_shape = net_input_shape	# Network Input
    self.replay_mem_size = replay_mem_size	# Replay Start Size
    self.minibatches = minibatches		# Minibatch Size
    self.gamma = gamma				# Discount Rate
    self.epsilon = epsilon			# Exploration Rate
    self.epsilon_decay = epsilon_decay		# Linear Exploration Decay
    self.epsilon_min = epsilon_min		# No lower Exploration Rate
    self.replay_mem = []			# The actual Replay Memory
    self.num_trains = 0				# The number of training steps

    # DQN
    self.dqn = DeepQNetwork(self.actions, self.net_input_shape, self.minibatches, self.gamma, DQN_WEIGHTS_PATH)

    # Target DQN
    self.target_dqn = DeepQNetwork(self.actions, self.net_input_shape, self.minibatches, self.gamma, TARGET_WEIGHTS_PATH)

    # Initialize DQN and Target DQN with the same weights, in case of loading
    self.target_dqn.model.set_weights(self.dqn.model.get_weights())
  ## END __INIT__ ##

  
  # Given a state, checks a random number against exploration rate
  # to determine if action is random or if this goes by max q value
  def act(self, state):
    if random.random() < self.epsilon:
      return random.randint(0, self.actions - 1)
    else:    
      q_vals = self.dqn.predict_model(state)
      return np.argmax(q_vals)
  ## END ACT ##

  
  # Provided the current state, action, reward, the next state, and if this was a last state
  # Add the whole experience to the replay memory. Phase out old memories as needed.
  def remember(self, current_state, action, reward, next_state, final):
    # phase out old memories in favor of new ones. Tell it to ignore when it was bad
    if len(self.replay_mem) >= REPLAY_MEM_SIZE:
      self.replay_mem.pop(0)
    # append a dictionary of the current_state, action, reward, next_state, and whether final
    self.replay_mem.append({'source':current_state, 'action':action, 'reward':reward, 'dest':next_state, 'final':final})
  ## END REMEMBER ##

  # Lowers the exploration rate linearly, to increase exploitation rate
  def change_epsilon(self):
    # If the change to epsilon will not put it below the min, decrease by the rate
    if self.epsilon - self.epsilon_decay > self.epsilon_min:
      self.epsilon = self.epsilon - self.epsilon_decay
    else:
      # otherwise, set epsilon to the desired minimum
      self.epsilon = self.epsilon_min
  ## ENG CHANGE_EPSILON ##

  # Provided the Agent itself, use minibatches to train the DQN
  def train(self):
    self.num_trains = self.num_trains + 1
    print("\nTraining #%d - epsilon: %f" % (self.num_trains, self.epsilon))
    batch = self.batch_create(MINIBATCH_SIZE)
    self.dqn.train_model(batch, self.target_dqn)
    return self.dqn
  ## CABOOSE ##  

  # Given a state, Provide the highsest q values
  def get_qs(self, state):
    # Get the Q values from the predict method
    q_vals = self.dqn.predict_model(state)
    # Take the spots where the Q values are highest out of the state
    spots = np.argwhere(q_vals == np.max(q_vals)).ravel()
    # Return a random choice of the spots to provide the maximum Q value
    return np.random.choice(spots)
  ## END GET_QS ##

  # Grabs a random state experience from the replay memory
  def get_random_state(self):
    return self.replay_mem[random.randrange(0, len(self.replay_mem))]['source']
  ## END GET_RANDOM_STATE ##

  # Given a size for the batch, Create one and return
  def batch_create(self, batch_size):
    batch = []
    i = 0
    for i in range(batch_size):
      batch.append(self.replay_mem[random.randrange(0, len(self.replay_mem))])
    ## END FOR ##
    return np.asarray(batch)
  ## END REPLAY ## 

  # Given the Agent, match the Target DQN's weights to the DQN's weights
  def reset_target_net(self):
    self.target_dqn.model.set_weights(self.dqn.model.get_weights())
  ## END RESET_TARGET_NET ##


#=========================<MAIN / OPERATIONS CODE>================================================
# Preprocess all images before performing any operations using them
# Take observation -> grayscale -> resize -> NP.Asarray 
def preprocess_obs(observations):
  # Take in the observations and create an image of RGB
  img = Image.fromarray(observations, 'RGB')
  # Convert image to grayscale
  img = img.convert('L')
  # resize the image
  img = img.resize(IMG_SIZE)
  return np.asarray(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0])
## END PREPROCESS_OBS ##

# Puts it all together
def run():
  scores = []	# Scores from each episode
  clipped_totals = [] # Clipped Reward Totals for each episode
  mean_q = []	# Average Q Value for each episode
  states = []	# Collection of States from to test from 

  print("Creating environment")
  env = gym.make(GAME)			# Creates the Atari environment
  net_input_shape = (4, 110, 84)	# Sets the input_shape
  # Create an Agent
  agent = QNetAgent(env.action_space.n, net_input_shape, replay_mem_size=REPLAY_MEM_SIZE, minibatches=MINIBATCH_SIZE, gamma=GAMMA, epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN)
  
  episode = 0			# Current Episode
  frame_tracker = 0		# Tracks number of Frames
  print("Beginning Deep Q Learning with Experience Replay")
  while episode < NUM_EPISODES:	# While we are beneath the number of episodes we wish to run
    print("#############\n")
    print("Episode %d\n" % (episode + 1))
    print("#############\n")

    score = 0			# Atari game's score for this episode
    clipped_total = 0		# Total from clipped rewards
    scores.append(score)	# Add a position to scores
    clipped_totals.append(clipped_total) # Add a position to clipped_totals
    mean_q.append(0)		# Add a position to mean_q
    observations = preprocess_obs(env.reset())	# preprocess the observation

    # Start the first state with the same observations
    c_state = np.array([observations, observations, observations, observations])

    time = 0	# time to determine episode length
    frame_tracker = frame_tracker + 1

    while time < MAX_LENGTH_FOR_EPISODE:	# As long as we don't take too long
      # Render the game
      env.render()
      # Select an action according to the current state
      action = agent.act(np.asarray([c_state]))

      # Take in observations, reward from action, if that ended the episode, and 
      observations, reward, final, info = env.step(action)
      # reprocess the observations
      observations = preprocess_obs(observations)
      # Create the Next State
      n_state = np.append(c_state[1:], [observations], axis=0)

      frame_tracker = frame_tracker + 1

      # Clip the reward for a positive/negative action for consistency across all games
      clip_rew = np.clip(reward, -1, 1)
      agent.remember(np.asarray([c_state]), action, clip_rew, np.asarray([n_state]), final)
     
      # If this was the last frame 
      if final or time == MAX_LENGTH_FOR_EPISODE - 1:
        break      

      # Train the Agent every 4 frames (time % 4), FREQUENCY is 4
      if time % FREQUENCY == 0:
	# If we have enough experience in the replay memory
        if len(agent.replay_mem) >= REPLAY_START_SIZE:
          # Train the agent!
          agent.train()
          
          # If the number of trains is nonzero and a multiple of 10,000, then reset target net 
          if agent.num_trains % TARGET_NET_UPDATE_FREQ == 0 and agent.num_trains >= TARGET_NET_UPDATE_FREQ:           
            agent.reset_target_net()

          # Change the epsilon according to linear decay
          agent.change_epsilon()
    
      # Update the current state to become the next state
      c_state = n_state
      # Adjust the score
      score = score + reward
      # Adjust the clipped total
      clipped_total = clipped_total + clip_rew      
      
      # Increase time for this episode
      time = time + 1
    ## END WHILE EPISODE LENGTH ##

    # Pull a fixed number of random states from the agent
    for i in range(MAX_STATES_COMP):
      states.append(agent.get_random_state())
    
    # Update values from the episode
    scores[episode] = score
    clipped_totals[episode] = clipped_total
    # Create the Q's for the states from the max Q values for each state
    test_qs = [agent.get_qs(state) for state in states]
    # Average the max Q values from the randomly chosen states
    max_q_overall = (np.max(test_qs))
    mean_q[episode] = (np.mean(test_qs))
    # Update to terminal
    print("Episode %d Length: %d, Score: %d, Clipped_Total: %d, Mean_Q: %d\n" % ((episode + 1), time, scores[episode], clipped_total,  mean_q[episode])) 
    print("Max Q: %d" % max_q_overall)

    episode = episode + 1

    # Unload the states for the next iteration
    for i in range(MAX_STATES_COMP):
      states.pop(0)
      
  ## END WHILE EPISODES ##


  ##### Save weights of network #####
  agent.dqn.model.save_weights(DQN_WEIGHTS_PATH)
  agent.target_dqn.model.save_weights(TARGET_WEIGHTS_PATH)

  file = open('avg_scores_and_qs.txt', 'w')
  for i in range(len(scores)):
    file.write("Episode %d Score: %d, Clipped_Total: %d, Mean_Q: %d\n" % ((i + 1), scores[i], clipped_totals[i], mean_q[i]))

# Main function
def main():
  run()
  

if __name__ == '__main__':
  main()
