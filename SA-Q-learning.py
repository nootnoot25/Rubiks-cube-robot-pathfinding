import time

import gym
import random
import rubiks_cube_gym
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#scramble denotes the moves used to move the cube away from its initial state, use "R", "F" and "U" to add moves
scramble = "R F"

#the following values keep track of the number of times the solution is reached by the heuristic, and by heuristic-Q
#the solution list is used to determine the avg, best and worst solutions
solutionreached_sa = 0
solutionreached_q = 0
solutionlist = []

#this function determines the fitness of a solution by seeing how far it is from the goal state string
def fitness_evaluation(cube_state, goal_state):

    if cube_state is None:
        return float('inf')  # Return a high fitness if the state is invalid
    fitness = 0

    for i in range(len(env.cube_reduced)):
        if env.cube_reduced[i] != goal_state[i]:
            fitness += 1

    return fitness

#update the q_table with values
def update_q_table(q_table, state, action, reward, learning_rate, discount_factor):
    current_q_value = q_table[state][action]
    updated_q_value = current_q_value + learning_rate * (reward - current_q_value)
    q_table[state][action] = updated_q_value

#perform simulated annealing
def simulated_annealing(goal_state, max_steps, temperature, cooling_rate):
    current_state = env.cube_reduced
    best_state = current_state
    #call fitness evaluator
    current_fitness = fitness_evaluation(current_state, goal_state)
    best_fitness = current_fitness

    #start with a a reward of 0
    reward_total = 0

    for step in range(max_steps):
        env.reset(scramble=scramble)
        action = random.randint(0, 2)

        # As the algorithm iterates through the search space, get the rewards and use them to populate the q-table
        current_state1 = env.cube_state
        # Simulate the cube state change with the action
        observation, reward, done, info = env.step(action)
        reward_total = reward_total + reward  # Accumulate the reward over time
        next_state = env.cube_reduced  # Get the next state
        next_state1 = env.cube_state

        # Calculate the fitness for the next state
        next_fitness = fitness_evaluation(next_state, goal_state)

        # Calculate the reward based on fitness (you can customize this)
        reward = 2.0 / (next_fitness + 1)

        # Update the Q-table with the calculated reward
        update_q_table(q_table, current_state1, action, reward, learning_rate, discount_factor)

        # Decide whether to move to the next state
        if next_fitness < current_fitness or random.random() < math.exp((current_fitness - next_fitness) / temperature):
            current_state = next_state
            current_fitness = next_fitness

            if current_fitness < best_fitness:
                best_state = current_state
                best_fitness = current_fitness

        if env.cube_reduced == goal_state:
            #print("Solution found! SA")
            global solutionreached_sa
            solutionreached_sa = solutionreached_sa + 1
            update_q_table(q_table, current_state1, action, reward_total, learning_rate, discount_factor)

        # Update the temperature
        temperature *= cooling_rate

    # Update the q_table with the values from the best solution found
    update_q_table(q_table, env.cube_state, action, reward_total, learning_rate, discount_factor)

    return best_state

# Perform Q-learning using the pre-populated q-table
def q_learning_to_solution(env, goal_state, max_episodes, max_tillreset, learning_rate, discount_factor, epsilon, q_table):

    for episode in range(max_episodes):
        state = env.reset(scramble=scramble)
        done = False
        episode_steps = 0  # Track the number of steps in this episode

        while not done and episode_steps < max_tillreset:
            # If wanted the enviroment can b rendered
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
                # Notify is a solution is found, also print the cube state so that the solve can be verified
                print("Solution found! - Q-LEARNING")

                global solutionreached_q
                global solutionlist

                solutionreached_q = solutionreached_q + 1
                solutionlist.append(episode_steps)

                print(env.cube_reduced)
                print(episode_steps)
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


env = gym.make('rubiks-cube-222-v0')

# Define the goal state (customize as needed)
goal_state = "WWWWOOGGRRBBOOGGRRBBYYYY"

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.9
max_episodes = 10000
max_tillreset = 50
epsilon = 0.1

# SA parameters
max_steps = 100000
temperature = 10
cooling_rate = 0.999


#Perform the experiment as many times as needed
for _ in range(1):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))

    result_state = simulated_annealing(goal_state, max_steps, temperature, cooling_rate)

    q_table_preq = q_table

    q_learning_to_solution(env, goal_state, max_episodes, max_tillreset, learning_rate, discount_factor, epsilon, q_table)

# Print results
print(f'Times solution found with SA: {solutionreached_sa}')
print(f'Times solution found with Q after SA: {solutionreached_q}')

highest_value, lowest_value, average_value = analyze_list(solutionlist)

print(f'Worst solution: {highest_value} steps')
print(f'Best solution: {lowest_value} steps')
print(f'Average solution: {average_value} steps')

env.close
