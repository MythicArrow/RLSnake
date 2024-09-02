import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.nn import functional as F

# Initialize Pygame
pygame.init()
width, height, cell_size = 400, 400, 20
win = pygame.display.set_mode((width, height))

# Define the Snake game environment with enhancements
class SnakeEnv:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.snake = [(width // 2, height // 2)]
        self.direction = (0, 0)
        self.food = self.place_food()
        self.done = False
        self.score = 0
        return self.get_state()
    
    def place_food(self):
        return (random.randint(0, (width - cell_size) // cell_size * cell_size),
                random.randint(0, (height - cell_size) // cell_size * cell_size))
    
    def get_state(self):
        state = np.zeros((width // cell_size, height // cell_size), dtype=np.float32)
        for segment in self.snake:
            state[segment[0] // cell_size, segment[1] // cell_size] = 1
        state[self.food[0] // cell_size, self.food[1] // cell_size] = 2
        return np.concatenate([state.flatten(),
                               [self.direction[0], self.direction[1]],
                               [self.snake[0][0], self.snake[0][1]]])

    def step(self, action):
        if action == 0: self.direction = (0, -cell_size)  # Up
        elif action == 1: self.direction = (0, cell_size)  # Down
        elif action == 2: self.direction = (-cell_size, 0) # Left
        elif action == 3: self.direction = (cell_size, 0)  # Right
        
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        if new_head in self.snake or new_head[0] < 0 or new_head[1] < 0 or new_head[0] >= width or new_head[1] >= height:
            self.done = True
            return self.get_state(), -10, self.done  # Penalty for collision
        
        self.snake = [new_head] + self.snake[:-1]
        if new_head == self.food:
            self.snake.append(self.snake[-1])
            self.food = self.place_food()
            self.score += 1
            reward = 10
        else:
            reward = 0
        
        return self.get_state(), reward, self.done
    
    def render(self):
        win.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(win, (0, 255, 0), (*segment, cell_size, cell_size))
        pygame.draw.rect(win, (255, 0, 0), (*self.food, cell_size, cell_size))
        pygame.display.update()

# Define the Q-Network with Dueling Architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)
        self.fc5 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc4(x)
        advantages = self.fc5(x)
        return value + (advantages - advantages.mean())

# Define the DQN Agent with enhancements
class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=0.001, buffer_size=10000, batch_size=64):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.q_network(state)).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Training Loop
env = SnakeEnv()
state_dim = len(env.get_state())
action_dim = 4
agent = DQNAgent(state_dim, action_dim)

num_episodes = 1000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
target_update_freq = 10

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward
        env.render()  # Render the game
        pygame.time.delay(100)  # Adjust delay for game speed

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % target_update_freq == 0:
        agent.update_target_network()

    print(f'Episode {episode}, Total Reward: {total_reward}')

pygame.quit()
