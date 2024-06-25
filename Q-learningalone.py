import gym
import random
import numpy as np
import rubiks_cube_gym
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Define your Gym environment
env = gym.make('rubiks-cube-222-v0')

#scramble denotes the moves used to move the cube away from its initial state, use "R", "F" and "U" to add moves
scramble="R"

#the following values keep track of the number of times the solution is reached by the heuristic, and by heuristic-Q
#the solution list is used to determine the avg, best and worst solutions
solutionreached_q = 0
solutionlist = []

# Solved state of the cube
goal_state = "WWWWOOGGRRBBOOGGRRBBYYYY"

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.9
max_episodes = 10000
max_tillreset = 50
epsilon = 0.1

# Q table
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

def fitness_evaluation(cube_state, goal_state):

    # Calculate the fitness of the cube state based on how many cubes are out of place.

    if cube_state is None:
        return float('inf')  # Return a high fitness if the state is invalid
    fitness = 0

    for i in range(len(env.cube_reduced)):
        if env.cube_reduced[i] != goal_state[i]:
            fitness += 1

    return fitness

# Update q table
def update_q_table(q_table, state, action, reward, learning_rate, discount_factor):
    current_q_value = q_table[state][action]
    updated_q_value = current_q_value + learning_rate * (reward - current_q_value)
    q_table[state][action] = updated_q_value

# Perform Q-learning
def q_learning_to_solution(env, goal_state, max_episodes, max_tillreset, learning_rate, discount_factor, epsilon, q_table):

    for episode in range(max_episodes):
        state = env.reset(scramble=scramble)
        done = False
        episode_steps = 0  # Track the number of steps in this episode

        while not done and episode_steps < max_tillreset:
            # Choose an action using epsilon-greedy strategy
            #env.render()
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit


            next_state, reward, done, _ = env.step(action)

            current_fitness = fitness_evaluation(state, goal_state)
            next_fitness = fitness_evaluation(next_state, goal_state)
            reward = 2.0 / (next_fitness + 1)

            # Update the Q-table using the Q-learning update rule
            update_q_table(q_table, state, action, reward, learning_rate, discount_factor)
            state = next_state
            episode_steps += 1

            if done == True:
                print("Solution found! - Q-LEARNING")
                print(env.cube_reduced)
                print(episode_steps)

                global solutionreached_q
                global solutionlist

                solutionreached_q = solutionreached_q + 1
                solutionlist.append(episode_steps)
                break

def analyze_list(numbers):
    # Check if the list is not empty
    if not numbers:
        return None, None, None

    # Sort the list in ascending order
    sorted_numbers = sorted(numbers)

    # Find the highest and lowest values
    highest = sorted_numbers[-1]
    lowest = sorted_numbers[0]

    # Calculate the average
    average = sum(sorted_numbers) / len(sorted_numbers)

    return highest, lowest, average


for _ in range(10):
    q_learning_to_solution(env, goal_state, max_episodes, max_tillreset, learning_rate, discount_factor, epsilon, q_table)

# Print results
print(f'Times solution found with Q: {solutionreached_q}')

highest_value, lowest_value, average_value = analyze_list(solutionlist)

print(f'Worst solution: {highest_value} steps')
print(f'Best solution: {lowest_value} steps')
print(f'Average solution: {average_value} steps')