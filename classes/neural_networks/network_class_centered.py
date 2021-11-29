import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy
import numpy as np
import pandas as pd
import sys, os, scipy.signal
import zipfile

from RNNs import recurrent_networks
from Bandits import bandit_class as bc
from Bandits import fixed_bandit_class as fbc
from Helpers.helpers import dot2_
from Helpers.helpers import zip2csv


"""

Neural Network Class

methods: 
    
    train

        Input: 
            bandit 
            number of hidden neurons
            number of samples
            noise = bol
            weber_fraction
            entropy_scaling
            path_to_save_model
            path_to_save_progress
            
        Output:
            save model
            track progress on tensorboard
            
    test
    
        Input:
            bandit
            n_replications
            path_to_save
            
        Output: 
            save .csv files and compress
"""


#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# class to generate the bandit tasks
class conditioning_bandit():
    def __init__(self, game):
        self.game = game        
        self.reset()
        
    def set_restless_prob(self):
        self.bandit         = self.restless_rewards[self.timestep]
        
    def reset(self):
        self.timestep          = 0 
        rewards , reward_probs = self.game.generate_task()     
        self.restless_rewards  = rewards
        self.reward_probs  = reward_probs
        self.set_restless_prob()
        
    def pullArm(self,action):
        if self.timestep >= (len(self.restless_rewards) - 1): done = True
        else: done = False
        return self.bandit[int(action)], done, self.timestep

    def update(self):
        self.timestep += 1
        self.set_restless_prob()

# class to define the graph
class AC_Network():
    def __init__(self, trainer, noise, coefficient
                 , n_hidden_neurons, n_arms):
        '''
        Returns the graph. 
        Takes as input: trainer, a tensorflow optimizer
                        noise, with computation noise (noise=1) or decision entropy (noise=0)
                        coefficient, coefficient for the computation noise or decision entropy

        '''
        if noise : regularize = 0
        else: regularize = coefficient
            
        # Input
        self.prev_rewardsch        = tf.placeholder(shape=[None,1], dtype=tf.float32)
        self.prev_actions          = tf.placeholder(shape=[None], dtype=tf.int32)
        self.prev_actions_onehot   = tf.one_hot(self.prev_actions, n_arms, dtype=tf.float32) #changed
        self.timestep              = tf.placeholder(shape=[None,1], dtype=tf.float32)
        input_                     = tf.concat([self.prev_rewardsch, self.prev_actions_onehot],1)

        self.actions             = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot      = tf.one_hot(self.actions, n_arms, dtype=tf.float32) #changed        

        #Recurrent network for temporal dependencies
        nb_units = n_hidden_neurons
        lstm_cell       = recurrent_networks.RNN(n_arms+1, nb_units, noise) # input (last reward, one-hot actions)
        h_init          = np.zeros((1, nb_units), np.float32)
        self.state_init = [h_init]        
        self.h_in       = tf.placeholder(tf.float32, [1, nb_units])        
        self.h_noise    = tf.placeholder(tf.float32, [None, nb_units])        
        self.state_in   = self.h_in
        all_noises      = self.h_noise

        if noise: 
            all_inputs         = tf.concat((input_, all_noises), axis=1)
            rnn_in             = tf.transpose(tf.expand_dims(all_inputs, [0]),[1,0,2])
        else:
            rnn_in = tf.transpose(tf.expand_dims(input_, [0]),[1,0,2])
        
        states, self.added_noises_means    = tf.scan(lstm_cell.step, rnn_in, initializer=(self.state_in, 0.))
        self.states_means                  = tf.reduce_mean(tf.math.abs(states), axis=(-1,-2))
        
        lstm_h         = states[:,0]
        self.state_out = states[:1,0]
        rnn_out        = lstm_h

        self.policy = slim.fully_connected(rnn_out, n_arms, activation_fn=tf.nn.softmax,
            weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)   #changed      
        
        #Get ops for loss functions and gradient updating.
        self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                
        self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
        
        #Loss functions
        self.entropy     = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
        
        # added Reinforce with Baseline (mean of discounted returns)
        self.baseline = tf.reduce_mean(self.advantages)
        self.policy_loss = - tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7) * (self.advantages - self.baseline))
        print('regularization is {0}'.format(regularize))
        self.loss        = self.policy_loss - self.entropy * regularize
        
        self.loss_entropy = self.entropy * regularize

        #Get gradients from network using losses
        local_vars            = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gradients        = tf.gradients(self.loss,local_vars)
        # added 
        self.gradient_norm = tf.global_norm(self.gradients)
        
        self.var_norms        = tf.global_norm(local_vars)         
        self.apply_grads      = trainer.apply_gradients(zip(self.gradients,local_vars))
        
class Worker():
    def __init__(self, game, trainer, model_path, coefficient, model_name, noise
                 , path_to_save_progress, n_hidden_neurons, n_arms, num_steps,
                 n_iterations):
        
        self.model_path            = model_path
        self.trainer               = trainer
        self.episode_rewards       = []
        self.episode_lengths       = []
        self.addnoises_mean_values = []
        self.hidden_mean_values    = []
        self.episode_reward_reversal = []
        model_name                += '_{}'.format('RNN')
        self.summary_writer        = tf.summary.FileWriter(path_to_save_progress + str(model_name))
        self.coefficient    = coefficient
        self.ac_network = AC_Network(trainer, noise, coefficient, n_hidden_neurons, n_arms)
        self.env      = game
        self.num_steps = num_steps
        self.n_iterations = n_iterations
        self.n_hidden_neurons = n_hidden_neurons
        self.n_arms = n_arms
        
    def train(self, rollout, sess, gamma, bootstrap_value):
        '''
        train method
        '''        
        rollout           = np.array(rollout)
        actions           = rollout[:,0]
        rewards_ch        = rollout[:,1]
        timesteps         = rollout[:,2]
        h_noises          = rollout[:,3]

        prev_actions      = [2] + actions[:-1].tolist()    # initialize one-hot vector representing the previous chosen action of episode to 0
        prev_rewards_ch   = [0] + rewards_ch[:-1].tolist() # initialize previous observed reward of episode to 0
        
        self.rewards_plus  = np.asarray(rewards_ch.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]

        rnn_state = self.ac_network.state_init[0]
        feed_dict = {self.ac_network.prev_rewardsch:np.vstack(prev_rewards_ch), self.ac_network.prev_actions:prev_actions,
            self.ac_network.h_noise:np.vstack(h_noises),                     
            self.ac_network.actions:actions, self.ac_network.timestep:np.vstack(timesteps),
            self.ac_network.advantages:discounted_rewards, self.ac_network.h_in:rnn_state}            
        # added grad_
        p_l,e_l,v_n,_, grad_ = sess.run([self.ac_network.policy_loss,
            self.ac_network.entropy,
            self.ac_network.var_norms,
            self.ac_network.apply_grads, 
            self.ac_network.gradient_norm],
            feed_dict=feed_dict)
        
        # added grad_
        return p_l / len(rollout),e_l / len(rollout), 0.,v_n, grad_
    
    def work(self, gamma, sess, saver, train):
        '''
        This is the main function
        Takes as input: gamma, the discount factor
                        sess, a Tensorflow session
                        saver, a Tensorflow saver
                        train boolean, do we train or not?
        The function will train the agent on the A task. To do so, the agent plays an A episode, and at the end of the episode, 
        we use the experience to perform a gradient update. When computation noise is assumed in the RNN, the noise realizations are 
        saved in the buffer and then fed to the back-propagation process.
        '''
        
        #################################################################################
        #################### if network is tested create a dataframe ####################
        if train == False:
            
            # prepare colnames for p_rew, softmax, unit activity and unit noise
            rnn_state_col_names = ['']*self.n_hidden_neurons
            rnn_state_noise_col_names = ['']*self.n_hidden_neurons
            
            rnn_prob_rew = ['']*self.n_arms
            rnn_softmax = ['']*self.n_arms
            
            for i in range(self.n_hidden_neurons):
                rnn_state_col_names[i] = 'rnn_state_'+str(i+1)
                rnn_state_noise_col_names[i] = 'added_noise_rnn_state_'+str(i+1)
                
            for i in range(self.n_arms):
                rnn_prob_rew[i] = 'p_rew_'+str(i+1)
                rnn_softmax[i] = 'softmax_'+str(i+1)           
            
            # prepare colname for bandit parameter
            if self.env.game.bandit_type == 'restless':
                # bandit_par = self.env.game.bandit_parameter  # have to change this
                bandit_par = 'sd_noise'
            if self.env.game.bandit_type == 'stationary':
                # bandit_par = self.env.game.bandit_parameter  # have to change this
                bandit_par = 'p_rew_best'
            else: 
                bandit_par = 'bandit parameter'
                
            colnames = [bandit_par,'choice', 'reward']
            
            colnames.extend(rnn_prob_rew + rnn_softmax + rnn_state_col_names + rnn_state_noise_col_names)
            
            df = pd.DataFrame(columns=colnames)
            
            # prepare variables to collect data
            my_a = []
            my_rch = []
            my_r_prob = np.zeros([self.num_steps, self.n_arms])
            rnn_softmax_arr = np.zeros([self.num_steps, self.n_arms])
            rnn_state_arr = np.zeros([ self.num_steps, self.n_hidden_neurons])
            rnn_state_noise_arr = np.zeros([self.num_steps, self.n_hidden_neurons])
        ##################################################################################
        
        episode_count = 0
        while True:
            episode_buffer, state_mean_arr, added_noise_arr = [], [], []
            episode_reward, episode_step_count = 0, 0
            d, a, t, rch       = False, 2, 0, 0 #initialization parameters (in particular, the previous action is initialized to a null one-hot vector, a=2)
            rnn_state          = self.ac_network.state_init[0]
            self.env.reset()
            
            while d == False:
                h_noise = np.array(np.random.normal(size=self.ac_network.state_init[0].shape) * self.coefficient, dtype=np.float32)
                #Take an action using probabilities from policy network output.
                feed_dict = {self.ac_network.prev_rewardsch:[[rch]], self.ac_network.prev_actions:[a],
                                self.ac_network.timestep:[[t]], self.ac_network.h_in:rnn_state, self.ac_network.h_noise:h_noise}
                                    
                a_dist,rnn_state_new,added_noise,state_mean = sess.run([self.ac_network.policy,self.ac_network.state_out,
                                                              self.ac_network.added_noises_means, self.ac_network.states_means], 
                                                              feed_dict=feed_dict)
                a                   = np.random.choice(a_dist[0],p=a_dist[0])
                a                   = np.argmax(a_dist == a)
                rnn_state           = rnn_state_new[:2]
                rch,d,t             = self.env.pullArm(a)
                # for tensorboard: if rewards 1, 0
                episode_reward     += rch
                # for tensorboard: if rewards 1, -1
                # episode_reward     += (rch + 1)/2.
                episode_step_count += 1
                state_mean_arr.append(state_mean)
                added_noise_arr.append(added_noise)
                episode_buffer.append([a,rch,t,h_noise,d])
                
                # if network is tested collect vaiables
                if train == False: 
                    my_a.append(a)
                    my_rch.append(rch)
                    my_r_prob[t] = self.env.reward_probs[t]
                    rnn_softmax_arr[t] = a_dist
                    rnn_state_arr[t] = rnn_state
                    rnn_state_noise_arr[t] = h_noise
                
                if not d:
                    self.env.update()        
            
            self.addnoises_mean_values.append(np.mean(added_noise_arr))
            self.hidden_mean_values.append(np.mean(state_mean_arr))
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_step_count)

            # Update the network using the experience buffer at the end of the episode.
            # added gg
            if len(episode_buffer) != 0 and train == True:
                p_l,e_l,g_n,v_n, gg = self.train(episode_buffer,sess,gamma,0.0)
                
            # stop after first trial if model is tested
            if train == False:
                
                # populate dataframe 
                df['choice'] = my_a
                df['reward'] = my_rch
                df[rnn_prob_rew] = my_r_prob
                df[rnn_state_col_names] = rnn_state_arr
                df[rnn_state_noise_col_names] = rnn_state_noise_arr
                df[rnn_softmax] = rnn_softmax_arr
                df[bandit_par] = self.env.game.bandit_parameter
                
                return df

            # Periodically save summary statistics.
            if episode_count != 0:
                if episode_count % 500 == 0 and train == True:
                    saver.save(sess, self.model_path+'/model-'+str(episode_count)+'.cptk')
                    print("Saved Model Episodes: {}".format(str(episode_count)))
                
                if train == True:
                    if episode_count % self.n_iterations == 0: # stopping criterion
                        return None

                mean_reward    = np.mean(self.episode_rewards[-50:])
                mean_noiseadd  = np.mean(self.addnoises_mean_values[-50:])
                mean_hidden    = np.mean(self.hidden_mean_values[-50:])
                # mean_reversal  = np.mean(self.episode_reward_reversal[-1])
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                # summary.value.add(tag='Perf/reversal_Reward', simple_value=float(mean_reversal))
                summary.value.add(tag='Info/Noise_added', simple_value=float(mean_noiseadd))
                summary.value.add(tag='Info/Hidden_activity', simple_value=float(mean_hidden))
                summary.value.add(tag='Parameters/biases_transition', simple_value=np.abs(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[3])).mean())
                summary.value.add(tag='Parameters/matrix_transition', simple_value=np.abs(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[1])).mean())                
                summary.value.add(tag='Parameters/matrix_input', simple_value=np.abs(sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[2])).mean())                                
                if train == True:
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    # added gg
                    summary.value.add(tag='Gradnorm', simple_value=float(gg))
                self.summary_writer.add_summary(summary, episode_count)
                self.summary_writer.flush()
                                
            episode_count += 1


class neural_network:
    
    def __init__(self 
                 , bandit
                 , noise = True
                 , weber_fraction = None
                 , entropy_scaling = None
                 , n_hidden_neurons = 48
                 , n_iterations = 50000
                 , path_to_save_model = 'Save_Models/'
                 , path_to_save_progress = 'Tensorboard/'):
        
        self.bandit = bandit
        self.noise = noise
        self.weber_fraction = weber_fraction
        self.entropy_scaling = entropy_scaling
        self.n_hidden_neurons = n_hidden_neurons
        self.n_iterations = n_iterations
        self.path_to_save_model = path_to_save_model
        self.path_to_save_progress = path_to_save_progress
        
        # define discount parameter
        self.gamma      = .5
        
        if self.noise:
            self.parameter = self.weber_fraction
        if not self.noise:
            self.parameter = self.entropy_scaling
    
        self.model_name = 'cent_RNN_N_{}_P_{}_{}_D_{}_P_{}_A_{}_N_{}_rew_0'.format(str(self.noise)[0]
                                                                , dot2_(self.parameter)
                                                                , self.bandit.bandit_type[0:4]
                                                                , str(self.bandit.dependant)[0]
                                                                , dot2_(self.bandit.bandit_parameter) 
                                                                , self.bandit.arms
                                                                , self.bandit.num_steps)
        
        self.model_path = self.path_to_save_model + self.model_name
        
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=1e-4) 
        
    def train(self):
        
        # train the RNN
        train = True
        
        # create folder to save models
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
                
        # create folder to save progress
        if not os.path.exists(self.path_to_save_progress):
            os.makedirs(self.path_to_save_progress)
        
        # create the graph
        self.worker  = Worker(conditioning_bandit(self.bandit)
                      , self.trainer, self.model_path, self.parameter
                      , self.model_name, self.noise
                      , self.path_to_save_progress
                      , self.n_hidden_neurons
                      , self.bandit.arms
                      , self.bandit.num_steps
                      , self.n_iterations)
        
        # create the saver
        self.saver   = tf.train.Saver(max_to_keep=5)
        
        # start tf.Session
        with tf.Session() as sess:
            
            print('Training Model: {}'.format(self.model_name))
            # initialise variables
            sess.run(tf.global_variables_initializer())
            # train
            self.worker.work(self.gamma,sess,self.saver,train)
            
        # reset the graph
        self.reset()
        
    def test(self, n_replications, bandit_param_range, bandit):
        
        # do not train the RNN
        train = False
        
        # create folder for test files
        if not os.path.exists('Test_Files'):
            os.makedirs('Test_Files')
            
        # create zip name to save test runs
        zip_name = '{}.zip'.format(self.model_name)
        
        # give zip file another name if we test presaved bandits (bandit = zip_file_name)
        if isinstance(bandit, str): 
            zip_name = 'Random_Walks_{}.zip'.format(self.model_name)
            # access zip file with saved bandit
            bandit_zip = zip2csv(bandit)
        
        # open zip file to save test runs
        with zipfile.ZipFile('Test_Files/{}'.format(zip_name), 'w', compression = zipfile.ZIP_DEFLATED) as my_zip:
        
            for par in bandit_param_range:
                
                if not isinstance(bandit, str):
                    # Use specified bandit with certain parameter
                    bandit.bandit_parameter = par                    
                    # assign bandit to RNN instance
                    self.bandit = bandit
                
                for run in range(n_replications):
                    
                    if isinstance(bandit, str):
                        # load presaved bandit
                        file_name = bandit[:-4]+'_P_{}_run_{}.csv'.format(dot2_(par), 
                                                                     run)
                        # extract presaved bandit
                        bandit_zip.extract_file(file_name)
                        
                        # load .csv to dataframe
                        df = pd.read_csv(file_name)
                                
                        # convert datafame into bandit class
                        self.bandit = fbc.load_bandit(df)
                        
                        # assign sd
                        self.bandit.bandit_parameter = par

                        # delete presaved bandit
                        bandit_zip.delete_file(file_name)
                        
                        # # assign bandit to RNN instance
                        # self.bandit = bandit
                        
                    
                    # create the graph
                    self.worker  = Worker(conditioning_bandit(self.bandit)
                      , self.trainer, self.model_path, self.parameter
                      , self.model_name, self.noise
                      , self.path_to_save_progress
                      , self.n_hidden_neurons
                      , self.bandit.arms
                      , self.bandit.num_steps
                      , self.n_iterations)
                    
                    # create saver
                    self.saver   = tf.train.Saver(max_to_keep=5)
                
                    with tf.Session() as sess:
                        print('Testing Model: {}'.format(self.model_name))
                        ckpt = tf.train.get_checkpoint_state(self.model_path)
                        self.saver.restore(sess,ckpt.model_checkpoint_path)
                        
                        # get test dataframe                       
                        df = self.worker.work(self.gamma,sess,self.saver,train)
                        file_name = self.model_name+'_test_{}_P_{}_run_{}.csv'.format(self.bandit.bandit_type[0:4]
                                                                                      , dot2_(self.bandit.bandit_parameter)
                                                                                  , str(run))
                        # create csv from dataframe
                        df.to_csv(file_name)
                        # write csv to zip file
                        my_zip.write(file_name)
                        # delete csv file
                        os.remove(file_name) 
                    
                    # reset graph
                    self.reset()

    def reset(self):
        tf.reset_default_graph()
        



