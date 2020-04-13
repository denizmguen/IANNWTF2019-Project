from tensorflow.keras.layers import Input, concatenate, Dense, BatchNormalization
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from tensorflow.keras.initializers import RandomUniform as uniform
from tensorflow.keras.activations import relu

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents. 
    Source: 'https://github.com/openai/spinningup/blob/master/spinup/algos/ddpg/ddpg.py'
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        # Our addition
        if len(obs.shape) > 1 or len(next_obs.shape) > 1:
            obs = np.reshape(obs, newshape=(-1,2))
        
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    done=self.done_buf[idxs])
					


class OrnsteinUhlenbeck():
    def __init__( self, action_dim=1,x0=None,mu=0.0, sigma=0.3, theta=0.15 ):
        self.action_dim = action_dim
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.x = x0
        self.dt = 1
        if x0 is None: self.reset() 
        
    
    def sample(self):
        """
        Calculates the next value of x depending on the last value of x and the 
        process' parameters.
        
        Returns:
            x: The next value sampled from a Ornstein Uhlenbeck Process
        """
        
        drift = self.theta*(self.mu-self.x)*self.dt
        dWt  = np.random.normal(0,1,size=self.action_dim) 
        dxt = drift + self.sigma*dWt
        self.x = self.x + dxt
        return self.x
    
    def reset(self):
        self.x = np.random.normal(loc=0,scale=self.sigma,size=self.action_dim)

class DDPGAgent():
    ''' 
    Initiates all subcomponents required for the DDPG algorithm 
    with standard parameter values as specified by (Lillicrap et al, 2015)
    '''
    
    def __init__(self,
                 env=None,
                 actor=None,
                 critic=None,
                 gamma=0.99,
                 tau=0.001,
                 batch_size=64,
                 replay_buffer_size=int(1e6),
                 noise_scale=0.3,
                 epsilon=0.999,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 l2_weight_decay=1e-2,
                 actor_nd1=400,
                 actor_nd2=300,
                 action_scale=1.0,
                 critic_nd1=400, 
                 critic_nd2=300,
                 overrepresent_goals=False,
                 q_modification=False):
        
        # Class Constants
        self.ENV = env
        self.GAMMA = gamma
        self.TAU = tau
        self.BATCH_SIZE = batch_size
        self.RP_BUFFER_SIZE = replay_buffer_size
        self.NOISE_SCALE = noise_scale
        self.EPSILON = epsilon # For reducing noise with increasing timesteps 
        
        #Environment Constants for convenience
        self.action_dim = env.action_space.shape[0]               
        self.observation_dim = env.observation_space.shape[0]  
        self.action_high = env.action_space.high
        self.action_low = env.action_space.low
        
        #Replaybuffer
        self.rp_buffer = ReplayBuffer(self.observation_dim, self.action_dim,
                                      self.RP_BUFFER_SIZE)
        #Remembering
        self.overrepresent_goals = overrepresent_goals
        self.q_modification = q_modification
        
        #Noise Process
        self.noise_process = OrnsteinUhlenbeck(self.action_dim, sigma=noise_scale)
        
        #Networks
        self.actor  = self.initialize_actor(actor_nd1,
                                            actor_nd2,
                                            action_scale) if actor == None else actor
        self.critic = self.initialize_critic(critic_nd1,
                                             critic_nd2) if critic == None else critic
        self.target_actor  =  self.initialize_target_actor()
        self.target_critic =  self.initialize_target_critic()
        
        #Optimizers
        self.optimizer_actor = tf.keras.optimizers.Adam(lr_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(lr_critic)
        self.l2_weight_decay = l2_weight_decay

        #For result visualization
        self.actor_losses = []
        self.critic_losses = []
        
        
    #=============================== ACTOR ========================================#
    def initialize_actor(self,nd1,nd2,action_scale):
        ''' Initializes a feedforward neural network that serves as a policy for this agent.'''
 
        # Initializers for different layers
        init_h1 = uniform(-1/np.sqrt(self.observation_dim), 1/np.sqrt(self.observation_dim))
        init_h2 = uniform(-1/np.sqrt(nd1) , 1/np.sqrt(nd1))
        init_out = uniform(-3e-3, 3e-3)
        
        # Input Layer
        x = Input(shape=(self.observation_dim,) ) # Dynamic input layer
        x_norm = BatchNormalization()(x)
        # First Hidden layer
        D1 = Dense(nd1,
                  activation=None,
                  kernel_initializer=init_h1,
                  bias_initializer=init_h1)(x_norm)
        D1_norm = BatchNormalization()(D1)
        D1_activation = relu(D1_norm)
        
        # Second Hidden layer
        D2 = Dense(nd2,
                  activation=None,
                  kernel_initializer=init_h2,
                  bias_initializer=init_h2)(D1_activation)
        D2_norm = BatchNormalization()(D2)
        D2_activation = relu(D2_norm)
        
        # Output Layer
        y = Dense(self.action_dim,
                  "tanh",
                  kernel_initializer=init_out,
                  bias_initializer=init_out)(D2_activation) 
        
        y = tf.multiply(y, action_scale)
        
        # Model Compilation
        actor = Model(inputs=x, outputs=y)
        actor.compile(optimizer='adam', loss="MSE") # these are placeholders, we fit the actor manually
        
        return actor
    
    def act(self,states,exploration=True):
        ''' Compute an action with optional noise
        
        params:
            state: (np.array) Batch of states with shape (batch_size, state_dim)
            exploration: (bool) Indicates wether noise should be added 
        returns:
            action: (np.array) Batch of actions with shape (batch_size, action_dim)
        '''  
        
        # Check wether states have the right shapes, should be (batch size, self.action_dim)
        if len(states.shape) == 1:
            states = np.expand_dims(states, axis=0)
            
        if states.shape[1] != self.observation_dim:
            states = states.reshape(-1, self.observation_dim)
        
        action = self.actor(states)
        
        # Add noise according to ornstein uhlenbeck process
        if exploration==True:
            noise = self.noise_process.sample()
            action += noise*self.EPSILON
            
        # Clip the action to the bounds given by the environment (usually unnecessary since
        # most environments have an action space of [-1,1] and tanh(x) -> [-1,1])
        
        action = np.clip(action, self.action_low, self.action_high) 
        
        return action

    #=============================== CRITIC =======================================#
    def initialize_critic(self, nd1, nd2):
        ''' Initialize a feedforward neural network that serves as the action-value function (or 'Q').''' 
        
        # Initializers for different layers
        init_h1 = uniform(-1/np.sqrt(self.observation_dim), 1/np.sqrt(self.observation_dim))
        init_h2 = uniform(-1/np.sqrt(nd1) , 1/np.sqrt(nd1))
        init_out = uniform(-0.001, 0.001)
        
        # Input Layer
        s = Input(shape=(self.observation_dim,))
        s_norm = BatchNormalization()(s)
        # First Hidden Layer
        D1 = Dense(nd1,
                  activation=None,
                  kernel_initializer=init_h1,
                  bias_initializer=init_h1)(s_norm)
        D1_norm = BatchNormalization()(D1)
        D1_activation = relu(D1_norm)
        
        # Second Hidden layer
        # Additional Input: Action
        a = Input(shape=(self.action_dim,))
        D1_a = concatenate([D1_activation,a], axis=1)
        
        D2= Dense(nd2,
                  "relu",
                  kernel_initializer=init_h2,
                  bias_initializer=init_h2)(D1_a)
        
        # Output Layer
        y = Dense(1,
                  "sigmoid",
                  kernel_initializer=init_out,
                  bias_initializer=init_out)(D2)
        
        # Model Compilation
        critic = Model(inputs=[s,a], outputs=y)
        critic.compile(optimizer="adam", loss="MSE", metrics=["mae"]) # Place Holders, we fit the critic manually
        return critic
        
    #=============================== TARGET NETWORKS ===============================#
    def initialize_target_actor(self):
        
        if "saved_model" in str(type(self.actor)):
            return None
        target_actor = tf.keras.models.clone_model(self.actor)
        target_actor.set_weights(self.actor.get_weights())
        return target_actor
    
    def initialize_target_critic(self):
        if "saved_model" in str(type(self.critic)):
            return None
        target_critic = tf.keras.models.clone_model(self.critic)
        target_critic.set_weights(self.critic.get_weights())
        return target_critic
    
    #=============================== UPDATES =======================================#
    
    def update(self):
        ''' Calls the update functions of all networks with a memory batch from the replay buffer. '''
        # Get a memory batch from the replay buffer
        batch = self.rp_buffer.sample_batch(self.BATCH_SIZE)
        states, actions, rewards, states2, dones = batch.values()
        
        # Convert memory batches (ndarrays) to tensors with 'cvt()' for gradienttape to work. 
        states = self.cvt(states)
        actions = self.cvt(actions)
        rewards = self.cvt(rewards)
        states2 = self.cvt(states2)
        dones = self.cvt(dones)
        
        # Update the networks
        self.update_critic(states, actions, rewards, states2, dones)
        self.update_actor(states)
        self.update_target_nets()

    def update_critic(self,states, actions, rewards, states2, dones):
        with tf.GradientTape() as tp:
            ''' The Bellman Update with MSE and L2 regularization'''
            # s,a
            critic_inputs = [states, actions]
            
            # Q(s,a) 
            critic_out = self.critic(critic_inputs)
            
            # P^(s')            (= a')
            target_actions = self.target_actor(states2)
            
            # Q^(s', P^(s'))   ( = Q^(s',a') )
            target_critic_out = self.target_critic([states2, target_actions])
            
            # y = r + gamma * (1-d) * Q^( s',  P^(s') )               
            critic_targets = rewards + self.GAMMA*(1-dones)*target_critic_out
            
            # MSE = 1/N * sum( ( y-Q(s,a) )² ) 
            mse = MSE(critic_targets,critic_out)
            
            # L2 = c*1/2*sum(weights**2)
            sum_weights_sq_per_layer = [tf.reduce_sum(v**2) for v in self.critic.trainable_variables 
                       if "bias" not in v.name]
            
            sum_weights_sq = tf.reduce_sum(sum_weights_sq_per_layer)
            
            l2 = self.l2_weight_decay * 1/2 * sum_weights_sq
            
            # Loss = MSE + L2 
            critic_loss = mse + l2
            self.critic_losses.append(critic_loss)
            
            # Apply Gradients
            critic_gradients = tp.gradient(critic_loss, self.critic.trainable_variables)
            self.optimizer_critic.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))  
            
        del tp
        
    def update_actor(self, states):
        ''' Update of the actor weights with with the scope of maximizing Q '''

        with tf.GradientTape(persistent=True) as tp: # Has to be persistent since we compute multiple gradients
            # P(s)      64 Values
            actions = self.actor(states)
            
            # Q(s,P(s)) 64 Values
            Qs = self.critic([states,actions])
            self.actor_losses.append(np.mean(Qs))
            
            # Compute the individual gradient vectors for each Q
            # list of 64 gradients, each gradient is a len-6-list: [(2,400),(400),(400,300),(300),(300,1),(1)],
            # because our actor has 3 layers and 3 biases.
            
            gradients = [tp.gradient(Q, self.actor.trainable_variables) for Q in Qs]
            # Yes, computing gradients inside a persistent gradient tape is inefficient but computing 
            # them outside just creates gradients with null entries.
            
            logging.debug(f'''gradients has self.batch_size entries:
                              {len(gradients) == self.BATCH_SIZE},
                              gradients[0] has as many entries as there are layers and biases in actor:
                              {len(gradients[0]) == len(self.actor.trainable_variables)}
                            '''
                         )
            
            
            
        # Neither a list nor a numpy array of tensors with different shapes is convertable
        # to a tensor by tensorflow, which is why we can't simply call tf_reduce_mean() on gradients.
        # Therefore, we will compute the mean of the gradients manually in
        # self.mean_of_gradients()    
        
        # Compute the sampled policy gradient
        mean_actor_gradient = self.mean_of_gradients(gradients)
        
        logging.debug(f'''Mean_actor gradient should have the following properties:
                        len == len(actor tv) : {len(mean_actor_gradient) == len(self.actor.trainable_variables)}
                        shape of G1 == shape of D1: {mean_actor_gradient[0].shape == gradients[0][0].shape}
                        No Nones : {not None in mean_actor_gradient[0].numpy()}
                        ''')
        
        # Apply Gradient
        # self.optimizer_actor is Adam. Adam optimizes the weights based on the negative gradient
        # to minimize the cost function (Q in this case). We want to maximize Q. Therefore, we flip
        # the sign of the mean gradient for each layer before we apply it. 
        mean_actor_gradient = [-g for g in mean_actor_gradient]
        self.optimizer_actor.apply_gradients(zip(mean_actor_gradient, self.actor.trainable_variables))    
        
        del tp
        
    def update_target_nets(self):
        '''Soft Update for Target networks'''
        aw = agent.actor.get_weights()           #actor weights
        cw = agent.critic.get_weights()          #critic weights
        taw = agent.target_actor.get_weights()   #target actor weights
        tcw = agent.target_critic.get_weights()  #target critic weights
        
        ntaw = [agent.TAU*w + (1-agent.TAU)*tw for w, tw in zip(aw,taw)] #new target actor weights
        agent.target_actor.set_weights(ntaw)
        
        ntcw = [agent.TAU*w + (1-agent.TAU)*tw for w, tw in zip(cw,tcw)] #new target critic weights
        agent.target_critic.set_weights(ntcw)

   
    def remember(self, state_transitions):
        ''' 
        Function to store a list of state_transitions
        '''
        # state_transitions[i] = [s_i,a_i,r_i,s'_i,d_i] 
        # 0:state, 1:action, 2:reward, 3:next_state, 4:done
        if type(state_transitions[0]) != list:
            state_transitions = [state_transitions]
            
        # Modify the rewards of the trajectory
        if self.q_modification:
            n_sts = len(state_transitions)
            # Modifiy the rewards of the states that preceeded it
            for i in range(2,min(50, n_sts)):
                state_transitions[-i][2] += state_transitions[-i+1][2] * self.GAMMA
                
        # if we want to overrepresent high reward states and the reward of the last state is very high:    
        if self.overrepresent_goals and state_transitions[-1][2] >= 50:
            # Store the sequence of states (or ONE state, depending on the value of "state_transitions")
            # ten times in the replay buffer to increase the probability that it is learned
            state_transitions *= 10
            
        # Add State transitions to replay buffer
        for state,action,reward,next_state,done in state_transitions:
            agent.rp_buffer.store(state.reshape(-1,self.observation_dim),
                                  action,
                                  reward,
                                  next_state.reshape(-1,self.observation_dim),
                                  done)
            
    #== Utils ==#
    def cvt(self,x):
        '''quickly converts arrays to tensors'''
        return tf.convert_to_tensor(x, dtype=tf.float64)
    
    def mean_of_gradients(self,gradients:list):
        ''' Takes a list of gradient lists and returns a list with the mean of all gradients'''
        
        nlayers = len(gradients[0])
        mean_gradients = list(np.zeros(nlayers)) # initialize a list with nlayers entries
        
        #For each layer i
        for i in range(nlayers):
            sum_i = 0
            #For all gradients that have been computed for this layer:
            for j in range(len(gradients)):
                sum_i += gradients[j][i]
            #Compute the average gradient for the ith layer
            avg_i = sum_i / len(gradients)
            mean_gradients[i] = avg_i
        
        return mean_gradients

def train_agent(agent,
                ENV,
                episodes=100,
                timesteps=800,
                train_interval=1,
                render=False,
                monitor=True,
                verbose=True,
                save_progress=True,
                save_interval=1,
                agent_name="",
                save_results=True,
                epsilon_decay=False,
                short_term_memory=1
               ):
    
    path = "saved_models\\" + agent_name + "\\"

    episodic_rewards = []

    for e in range(1, episodes+1):
        logging.info(f'Episode {e} out of {episodes}')
        
        if save_progress and e%save_interval==0:
            tf.saved_model.save(agent.actor,  export_dir = path+f"e{e}_actor")
            tf.saved_model.save(agent.critic, export_dir = path+f"e{e}_critic")

        rewards = []
        state_transitions = []
        state = env.reset()
        agent.noise_process.reset()
        
        for t in range(timesteps):
            if monitor and t%50==0:
                logging.info(f'episode {e}, timestep {t}')
            
            # Act
            try:
                action = agent.act(state, exploration=True)
            except:
                print(f"Bad state:  {state}")
                break
            # Observe
            next_state, reward, done, _ = env.step(action)
            if render: env.render()

            # Remember 
            state_transitions.append([state,action,reward,next_state,done])
            
            if t % short_term_memory == 0:
                agent.remember(state_transitions)
                state_transitions = []

            # Update
            if t % train_interval == 0 and agent.rp_buffer.size >= agent.BATCH_SIZE:
                agent.update()


            # (Update State)
            state = next_state
            # (Record Performance)
            rewards.append(reward)

            # End
            if done or t == timesteps-1:
                # Save Episode Score
                episodic_reward = sum(rewards)
                episodic_rewards.append(episodic_reward)
                # Print Info
                if verbose:
                    info = f" in a goal state after {t} steps " if done else " "
                    logging.info(f'Episode ended{info}with cumulative reward of:{episodic_reward}')
                    
                if epsilon_decay:
                    agent.EPSILON *= agent.EPSILON
                break

    env.close()
    
    episodic_rewards = pd.Series(episodic_rewards)
    actor_losses = pd.Series(agent.actor_losses)
    critic_losses = pd.Series(agent.critic_losses)
    if save_results:         
        episodic_rewards.to_csv(f"results/episodic_rewards/{agent_name}_e{str(episodes)}.csv")
        actor_losses.to_csv(f"results/losses/{agent_name}_e{str(episodes)}_actor.csv")
        actor_losses.to_csv(f"results/losses/{agent_name}_e{str(episodes)}_critic.csv")

    return episodic_rewards

def plot_results(episodic_rewards,
				agent_name="agent",
				save=True):
    plt.figure(figsize=(9, 3), dpi=100)
    plt.scatter(range(len(episodic_rewards)),episodic_rewards, lw=0.7, label="raw")
    plt.gca().set(title=f"Episodic rewards of {agent_name}",
                  xlabel="Episode",
                  ylabel="reward")
    plt.grid()
    if save: plt.savefig(f"results/episodic_rewards/{agent_name}")
    plt.show()

	
def test_agent(agent,
                ENV,
                episodes=10,
                timesteps=800,
                render=False,
                verbose=True,
                agent_name="",
                save_results=True,
                normalize=False,
                mu=0,
                sigma=1
              ):

    episodic_rewards = []

    for e in range(1, episodes+1):
        logging.info(f'Episode {e} out of {episodes}')
        rewards = []
        state = env.reset()
               
        for t in range(timesteps):
            action = agent.act(state, exploration=False)
            if normalize:
                action = (action-mu)/sigma
            state, reward, done,_ = env.step(action)
            if render: env.render()
            rewards.append(reward)
            if done or t == timesteps-1:
                # Save Episode Score
                episodic_reward = sum(rewards)
                episodic_rewards.append(episodic_reward)
                # Print Info
                if verbose:
                    info = f" in a goal state after {t} steps " if done else " "
                    logging.info(f'Episode ended{info}with cumulative reward of:{episodic_reward}')
                break

    env.close()
               
    episodic_rewards = pd.Series(episodic_rewards)
    if save_results:         
        episodic_rewards.to_csv(f"results/episodic_rewards/{agent_name}_e{str(episodes)}_test.csv")
    return episodic_rewards

	
if __name__ == "__main__":	
	''' Trains the Original DDPG Agent for 100 Episodes, plots and saves the results''' 
	tf.random.set_seed(100)
	np.random.seed(100)
	env = gym.make("MountainCarContinuous-v0") 
	agent_name = "original_ddpg" # We recommend renaming the agent, if you do not want to overwrite the original results.
	agent = DDPGAgent(env)
	episodic_rewards = train_agent(agent,env,
								   episodes=100,
								   timesteps=750,
								   train_interval=4,
								   render=True,
								   save_interval=10,
								   agent_name=agent_name)
	plot_results(episodic_rewards, agent_name)
	test_results=test_agent(agent,env,episodes=10)
	plot_results(test_results,agent_name+"_test")
	