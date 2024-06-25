import gym
import random
import numpy as np
import rubiks_cube_gym

# Define your Gym environment
env = gym.make('rubiks-cube-222-v0')

goal_state = "WWWWOOGGRRBBOOGGRRBBYYYY"

#scramble denotes the moves used to move the cube away from its initial state, use "R", "F" and "U" to add moves
scramble = "R F U R U R R F"

#the following values keep track of the number of times the solution is reached by the heuristic, and by heuristic-Q
#the solution list is used to determine the avg, best and worst solutions
solutionreached_sa = 0
solutionreached_q = 0
solutionlist = []


# Evaluate fitness of
def fitness_evaluation(cube_state, goal_state):
    if cube_state is None:
        return float('inf')
    fitness = 0
    for i in range(len(env.cube_reduced)):
        if env.cube_reduced[i] != goal_state[i]:
            fitness += 1
    return fitness

def update_q_table(q_table, state, action, reward, learning_rate, discount_factor):
    current_q_value = q_table[state][action]
    updated_q_value = current_q_value + learning_rate * (reward - current_q_value)
    q_table[state][action] = updated_q_value


def evaluate_individual(env, individual, goal_state):
    state = env.reset(scramble=scramble)
    for action in individual:

        reward_total=0

        current_state1 = env.cube_state
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
        if done:
            break
    return fitness_evaluation(current_state1, goal_state)

# Create a solution with a fixed length of steps
def create_individual(sequence_length):
    return [env.action_space.sample() for _ in range(sequence_length)]

# Perform solution crossover to generate new population
def crossover(parent1, parent2):
    crossover_point = len(parent1) // 2
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

# Mutate individuals to add randomness
def mutate(individual, mutation_rate):
    mutated_individual = []
    for action in individual:
        if np.random.rand() < mutation_rate:
            mutated_individual.append(env.action_space.sample())  # Mutate
        else:
            mutated_individual.append(action)
    return mutated_individual

# Perform the genetic algorithm
def genetic_algorithm(env, goal_state, population_size, generations, initial_mutation_rate, final_mutation_rate):
    sequence_length = 50
    population = [create_individual(sequence_length) for _ in range(population_size)]

    for generation in range(generations):
        fitness_scores = [evaluate_individual(env, individual, goal_state) for individual in population]

        # Elitism: Preserve the best individual
        best_index = np.argmin(fitness_scores)
        best_individual = population[best_index]

        # Update mutation rate dynamically
        current_mutation_rate = initial_mutation_rate

        for i in range(population_size):
            # Select parents based on fitness scores
            parents_indices = np.argsort(fitness_scores)[:2]
            parent1, parent2 = population[parents_indices[0]], population[parents_indices[1]]

            # Perform crossover to create offspring
            child = crossover(parent1, parent2)

            # Apply mutation to the child
            child = mutate(child, current_mutation_rate)

            # Replace the least fit individual in the population with the child
            worst_index = np.argmax(fitness_scores)
            population[worst_index] = child

        # Preserve the best individual in the population
        population[0] = best_individual

        # Print the best fitness in each generation
        best_fitness = min(fitness_scores)
        #print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        # Check if the goal state is reached
        #if env.cube_reduced == goal_state:

        if best_fitness == 0:
            print("Solution found! - Genetic algorithm")
            global solutionreached_sa
            solutionreached_sa = solutionreached_sa + 1
            break

# Set parameters for GA
population_size = 10
generations = 1000
initial_mutation_rate = 2
final_mutation_rate = 0.1

# Set parameters for Q-learning
learning_rate = 0.1
discount_factor = 0.9
max_episodes = 10000
max_tillreset = 50
epsilon = 0.1

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
                #print("Solution found! - Q-LEARNING")
                #print(env.cube_reduced)
                #print(episode_steps)

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

# Run the genetic algorithm
for _ in range(10):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))
    genetic_algorithm(env, goal_state, population_size, generations, initial_mutation_rate, final_mutation_rate)
    q_learning_to_solution(env, goal_state, max_episodes, max_tillreset, learning_rate, discount_factor, epsilon,
                           q_table)

# Print results
print(f'Times solution found with Genetic Algorithm: {solutionreached_sa}')
print(f'Times solution found with Q after Genetic Algorithm: {solutionreached_q}')

highest_value, lowest_value, average_value = analyze_list(solutionlist)

print(f'Worst solution: {highest_value} steps')
print(f'Best solution: {lowest_value} steps')
print(f'Average solution: {average_value} steps')

