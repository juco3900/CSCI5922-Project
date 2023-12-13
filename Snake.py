import numpy as np

class Snake:
    
    def __init__(self, grid_shape):
        self.grid_shape = grid_shape
        self.head_pos = [np.random.randint(2, grid_shape[0]-2), np.random.randint(2, grid_shape[1]-2)]
        shift_x, shift_y = 2*np.random.randint(2, size=2) - 1
        self.body_pos = [[self.head_pos[0], self.head_pos[1]+shift_x],
                         [self.head_pos[0]+shift_y, self.head_pos[1]+shift_x]]
        self.direction = 0
        self.watch_fruit = 0
        self.force_end_game = False
        self.timer = 0
        
        self.newFruit()
    
    def moveSnake(self):
        tail_pos = self.body_pos[-1]
        self.body_pos = [self.head_pos] + self.body_pos[:-1]
        if self.direction == 0: #up
            self.head_pos = [self.head_pos[0]-1, self.head_pos[1]]
        if self.direction == 1: #right
            self.head_pos = [self.head_pos[0], self.head_pos[1]+1]
        if self.direction == 2: #down
            self.head_pos = [self.head_pos[0]+1, self.head_pos[1]]
        if self.direction == 3: #left
            self.head_pos = [self.head_pos[0], self.head_pos[1]-1]
            
        if self.checkFruit():
            self.body_pos += [tail_pos]
            self.newFruit()
            self.timer = 0
        self.timer += 1
        if self.timer > 50:
            self.force_end_game = True
    
    def turnSnake(self, new_direction):
        if np.mod(self.direction - new_direction, 2) == 1:
            self.direction = new_direction
        
    def checkCollision(self):
        if self.head_pos[0] < 1 or self.head_pos[0] >= self.grid_shape[0] - 1:
            return True
        if self.head_pos[1] < 1 or self.head_pos[1] >= self.grid_shape[1] - 1:
            return True
        if self.head_pos in self.body_pos:
            return True
        return False
    
    def checkFruit(self):
        if self.head_pos == self.fruit_pos:
            self.watch_fruit = 1
            return True
        
    def newFruit(self):
        while True:
            self.fruit_pos = [np.random.randint(1, self.grid_shape[0]-1), np.random.randint(1, self.grid_shape[1]-1)]
            if self.fruit_pos != self.head_pos:
                if self.fruit_pos not in self.body_pos:
                    break
                    
    def gameState(self):
        #white background
        grid = np.ones(self.grid_shape+(3,), dtype=float)

        #blue head
        grid[self.head_pos[0], self.head_pos[1], 0] = 0
        grid[self.head_pos[0], self.head_pos[1], 1] = 0

        #green body
        for pos in self.body_pos:
            grid[pos[0], pos[1], 0] = 0
            grid[pos[0], pos[1], 2] = 0

        #red fruit
        grid[self.fruit_pos[0], self.fruit_pos[1], 1] = 0
        grid[self.fruit_pos[0], self.fruit_pos[1], 2] = 0

        #black walls
        grid[0,:,:] = grid[-1,:,:] = grid[:,0,:] = grid[:,-1,:] = 0

        return grid
    
    def snakeLength(self):
        return 1 + len(self.body_pos)
    
    def fruitDistance(self):
        return np.sqrt((self.head_pos[0] - self.fruit_pos[0])**2 + (self.head_pos[1] - self.fruit_pos[1])**2)