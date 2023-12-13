import tensorflow as tf
from tensorflow import keras

from Environment import *
from ReplayBuffer import *

class Actor:
    
    def __init__(self, grid_shape):
        self.gamma = 0.95
        self.minSampleSize = 64
        self.learning_rate = 0.001
        self.tau = 0.99
        
        self.dqn = keras.models.Sequential([
            keras.layers.Conv2D(input_shape=grid_shape+(3,), kernel_size=3, filters=4, padding='same', activation='relu'),
            keras.layers.Conv2D(kernel_size=5, filters=4, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(4, activation='softmax')
        ])
        self.target = keras.models.clone_model(self.dqn)

        self.MSE = keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.replayBuffer = ReplayBuffer()
    
    def save(self):
        self.dqn.save('Snake_main_model.keras')
        self.target.save('Snake_target_model.keras')
    
    def load(self):
        self.dqn = keras.saving.load_model('Snake_main_model.keras')
        self.target = keras.saving.load_model('Snake_target_model.keras')
        
    def push(self, state, action, next_state, reward, done):
        if reward != None:
            self.replayBuffer.push(state, action, next_state, reward, float(done))
    
    def optimumPolicy(self, qValue):
        return int(tf.argmax(qValue, axis=1))
    
    def randomPolicy(self, num_actions):
        return np.random.randint(num_actions)
    
    def proposeAction(self, state, epsilon):
        if np.random.uniform(0,1) > epsilon:
            qValue = self.dqn.call(state)
            return self.optimumPolicy(qValue)
        else:
            return self.randomPolicy(Environment.num_actions)
    
    def tdLoss(self, sample):
        states, actions, next_states, rewards, dones = sample
        qVals = tf.gather(self.dqn.call(states), actions, batch_dims=1)
        next_actions = tf.argmax(self.dqn.call(next_states), axis=1)
        next_qVals = tf.gather(self.target.call(next_states), next_actions, batch_dims=1)
        Y = rewards + self.gamma * (1 - dones) * next_qVals
        return self.MSE(qVals, Y)

    def learn(self):
        if self.replayBuffer.size() < self.minSampleSize:
            return 0
        sample = self.replayBuffer.sample()
        
        with tf.GradientTape() as tape:
            loss = self.tdLoss(sample)
        grads = tape.gradient(loss, self.dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.dqn.trainable_variables))
 
        for varInd in range(len(self.dqn.trainable_variables)):
            dqnParam, tgtParam = self.dqn.trainable_variables[varInd], self.target.trainable_variables[varInd]
            newParam = self.tau * tgtParam + (1 - self.tau) * dqnParam
            self.target.trainable_variables[varInd].assign(newParam)
   
        return loss