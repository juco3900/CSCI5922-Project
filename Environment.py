import numpy as np

from Snake import *

class Environment:
    
    num_actions = 4
    
    def __init__(self, grid_shape):
        self.step_ctr = 0
        self.snake = Snake(grid_shape)
    
    def reset(self):
        self.step_ctr = 0
        self.snake = Snake(self.snake.grid_shape)
        
    def state(self):
        return self.snake.gameState()
        
    def done(self):
        if self.snake.force_end_game:
            return True
        return self.snake.checkCollision()
    
    def reward(self):
        if self.snake.force_end_game:
            return None
        if self.done():
            return -1
        if self.snake.watch_fruit == 1:
            self.snake.watch_fruit = 0
            return 1
        return 1 / (4 * self.snake.fruitDistance())
    
    def applyAction(self, action):
        self.snake.turnSnake(action)
        self.snake.moveSnake()
        self.step_ctr += 1