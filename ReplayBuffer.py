import numpy as np
import tensorflow as tf

class ReplayBuffer:
    
    def __init__(self):
        self.memorySize = 50000
        self.batchSize = 256
        self.stepPtr = 0
        self.buffer = []
        
    def push(self, state, action, next_state, reward, done):
        step = (state, action, next_state, reward, done)
        if len(self.buffer) < self.memorySize:
            self.buffer += [step]
        else:
            self.buffer[self.stepPtr] = step
        self.stepPtr = (self.stepPtr + 1) % self.memorySize

    def sample(self):
        sampleSize = min(len(self.buffer), self.batchSize)
        samplePtr = -np.ones(sampleSize, dtype=int)
        
        if len(self.buffer) < 2 * self.batchSize:
            arr = np.arange(len(self.buffer), dtype=int)
            np.random.shuffle(arr)
            samplePtr = arr
        else:
            for i in range(sampleSize):
                r = None
                while True:
                    r = np.random.randint(len(self.buffer)) #len - 1
                    if r not in samplePtr[:i]:
                        break
                samplePtr[i] = r
        
        states, actions, next_states, rewards, dones = [], [], [], [], []
        for i in range(sampleSize):
            states += [self.buffer[samplePtr[i]][0]]
            actions += [self.buffer[samplePtr[i]][1]]
            next_states += [self.buffer[samplePtr[i]][2]]
            rewards += [float(self.buffer[samplePtr[i]][3])]
            dones += [float(self.buffer[samplePtr[i]][4])]
        
        sample = (tf.concat(states, axis=0), tf.convert_to_tensor(actions),
                  tf.concat(next_states, axis=0), tf.convert_to_tensor(rewards),
                  tf.convert_to_tensor(dones))
        return sample
    
    def size(self):
        return len(self.buffer)