import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate, Dense
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents. 
    'https://github.com/openai/spinningup/blob/master/spinup/algos/ddpg/ddpg.py'
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
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

					

class DDPGAgent():
    def __init__(self, env=None, actor=None, critic=None, gamma=0.99,
                 tau=0.005,batch_size=64,replay_buffer_size=int(10e6), noise_scale=0.3,epsilon=0.9):
        # Class Constants
        self.ENV = env
        self.GAMMA = gamma
        self.TAU = tau
        self.BATCH_SIZE = batch_size
        self.RP_BUFFER_SIZE = replay_buffer_size
        self.NOISE_SCALE = noise_scale
        self.EPSILON = epsilon # 
        
        #Just for convenience
        self.action_dim = env.action_space.shape[0]               
        self.observation_dim = env.observation_space.shape[0]  
        self.action_high = env.action_space.high
        self.action_low = env.action_space.low
        
        #Replaybuffer
        self.rp_buffer = ReplayBuffer(self.observation_dim, self.action_dim,
                                      self.RP_BUFFER_SIZE)
        # Networks
        self.actor  = self.initialize_actor()
        self.critic = self.initialize_critic()
        self.target_actor  =  self.initialize_actor()
        self.target_critic =  self.initialize_critic()
        self.optimizer = optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        # Dummy Inputs for actor loss
        self.dummy_Q_target_prediction_input = np.zeros((self.BATCH_SIZE, 1))
        self.dummy_dones_input = np.zeros((self.BATCH_SIZE, 1))
        
        #Progress Tracking
        self.actor_losses = []
        self.critic_losses = []
        
        
    #=============================== ACTOR ===============================#
    
    def initialize_actor(self):
        x = Input(shape=(self.observation_dim,))
        D1= Dense(32,"relu",
                    kernel_initializer='he_uniform',
                    bias_initializer="he_uniform")(x)
        D2= Dense(32,"relu",
                    kernel_initializer='he_uniform',
                    bias_initializer="he_uniform")(D1)
        
        y = Dense(1,"sigmoid",
                    kernel_initializer="he_uniform",
                    bias_initializer="he_uniform")(D2)
        
        actor = Model(inputs=x, outputs=y)
        actor.compile(optimizer='adam', loss="MSE") # does not matter anyway
        return actor
    

    def act(self,states,noise=None):
        """Returns an action (=prediction of local actor) given a state.
        Adds a gaussion noise for exploration. 
        params:
            :state: the state batch
            :noise: add noise. If None defaults self.ACT_NOISE_SCALE is used.
                    If 0 ist passed, no noise is added and clipping passed
        """   
        if len(states.shape) == 1: states = states.reshape(1,-1)   
        noise = np.random.normal(0,self.NOISE_SCALE, len(states))
        #states = self.cvt(states)
        action = self.actor(states)
        action += noise
        action = np.clip(action, self.action_low, self.action_high)
        
        return action

    
    #=============================== CRITIC ===============================#
    
    def initialize_critic(self):
        s = Input(shape=(self.observation_dim,))
        a = Input(shape=(self.action_dim,))
        x = concatenate([s,a], axis=1)
        D1= Dense(32,"relu",
                    kernel_initializer='he_uniform',
                    bias_initializer="he_uniform")(x)
        D2= Dense(32,"relu",
                    kernel_initializer='he_uniform',
                    bias_initializer="he_uniform")(D1)
        
        y = Dense(1,"sigmoid",
                    kernel_initializer="he_uniform",
                    bias_initializer="he_uniform")(D2)
        
        critic = Model(inputs=[s,a], outputs=y)
        critic.compile(optimizer="adam", loss="MSE", metrics=["mae"]) # Just for show, we update manually
        return critic
    
    
    
    
    #=============================== TARGET NETWORKS ===============================#
    def initialize_target_actor(self):
        target_actor = self.initialize_actor()
        target_actor.set_weights(self.actor.get_weights())
        return target_actor
    
    def initialize_target_critic(self):
        target_critic = self.initialize_critic()
        target_critic.set_weights(self.critic.get_weights())
        return target_critic
    
    
    
    #=============================== UPDATES ===============================#
    
    # Updates all subcomponents of the actor
    def update(self):
        batch = self.rp_buffer.sample_batch(self.BATCH_SIZE)
        states, actions, rewards, states2, dones = batch.values()
        
        # Convert memories to tensors 'cvt()' for gradienttape to work
        states = self.cvt(states)
        actions = self.cvt(actions)
        rewards = self.cvt(rewards)
        states2 = self.cvt(states2)
        dones = self.cvt(dones)

        self.update_critic(states, actions, rewards, states2, dones)
        self.update_actor(states)
        self.update_target_nets()
     
    
    def update_critic(self,states, actions, rewards, states2, dones):
        with tf.GradientTape() as tp:
            ''' The Bellman Update with MSE '''
            # s,a
            critic_inputs = [states, actions]
            
            # Q(s,a) 
            critic_out = self.critic(critic_inputs)
            
            # P^(s')
            target_actions = self.target_actor(states2)
            
            # Q^(s', P^(s'))
            target_critic_out = self.target_critic([states2, target_actions])
            
            # y = r + gamma * (1-d) * Q^( s',  P^(s') )               
            critic_targets = rewards + self.GAMMA*(1-dones)*target_critic_out
            
            # L2 = sum( (Q(s,a)-y)² ) / Batchsize
            critic_loss = MSE(critic_targets,critic_out)
            self.critic_losses.append(critic_loss)
            
            critic_gradients = tp.gradient(critic_loss, self.critic.trainable_variables)
            
            self.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))  
        
    def update_actor(self, states):
        with tf.GradientTape() as tp:
            ''' Gradient Update for the Policy in Continuous Action Space. '''
            # P(s)
            actions = self.actor(states)
            
            # Q(s,P(s))
            actor_target = self.critic([states,actions])
            
            # -mean(Q(s,P(s))) negative because Adam minimizes the loss
            # We want to maximize actor_target
            actor_target = -tf.reduce_mean(actor_target)
            self.actor_losses.append(actor_target)
            
            actor_gradients = tp.gradient(actor_target, self.actor.trainable_variables)
            
            self.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
    
    
    def update_target_nets(self):
        aw = agent.actor.get_weights()
        cw = agent.critic.get_weights()
        taw = agent.target_actor.get_weights()
        tcw = agent.target_critic.get_weights()

        naw = [agent.TAU*w + (1-agent.TAU)*tw for w, tw in zip(aw,taw)]
        agent.target_actor.set_weights(naw)
        
        ncw = [agent.TAU*w + (1-agent.TAU)*tw for w, tw in zip(cw,tcw)]
        agent.target_critic.set_weights(ncw)
        
    #== Utils ==#
    def cvt(self,x):
        '''quickly converts arrays to tensors'''
        return tf.convert_to_tensor(x, dtype=tf.float32)