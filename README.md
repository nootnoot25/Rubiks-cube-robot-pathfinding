# Rubiks-cube-robot-pathfinding
Exploring the limits of Q-learning guided by optimisation
heuristics to solve the pocket Rubik’s cube. A project done under the module "Autonomous Robotic Systems" at the University of Nottingham Malaysia


# Basis of the project
In this project, the rubiks cube is used as an enviroment to test how well different pathfinding algorithms (3-opt, Genetic algorithm and Simulated Annealing) are able to improve the abilities of Q-learning. Usually, reinforcement learning algorithms struggle with pathfinding problems as reward schemes are difficult to organise. As such, we try to use the path finding algorithms as a guiding method for reinforcement learning

# Methodology
The 2x2 Rubik's Cube, or pocket cube, consists of 3,674,160 possible states. It serves as an ideal complex environment for testing algorithms. Moves are described using letters (e.g., "F" for front, "R" for right). Our approach uses Markov Decision Processes (MDPs), where:

    States (S) capture cube configurations.
    Actions (A) represent allowable moves.
    Transition probabilities (P) handle state transitions.
    The reward function (R) encourages progress.

# Chosen Optimisation heuristics
Simulated Annealing (SA): A probabilistic optimization algorithm that explores neighboring solutions and accepts worse solutions with decreasing probability over time.

3-opt: A local search optimization algorithm improving solutions by swapping 3 edges.

Genetic Algorithm (GA): An evolutionary optimization technique evolving a population of potential solutions over multiple generations.


# Experimental Design
1. Set a scramble starting with one random move.
2. Run optimization heuristics and record the number of times the cube is solved.
3. Populate a Q-table with rewards from moves explored by heuristics.
4. Perform Q-learning for 10,000 episodes, each with a maximum of 50 moves.
5. Record the performance of Q-learning-heuristic pairs across 10 trials.
6. Increment the scramble complexity by adding a move and repeat until all algorithms fail to solve the cube.

# Results (Performance of the different algorithms)
Q-learning alone: Solved scrambles up to 3 moves.

Simulated Annealing (SA): Enhanced Q-learning to solve up to 8-move scrambles.

3-opt: Solved scrambles consistently and improved Q-learning performance up to 7 moves.

Genetic Algorithm (GA): Enhanced Q-learning performance up to 7 moves.

# Future Work
Further research should explore combining optimization heuristics and investigating deep-Q learning to push the boundaries of problem complexity that can be addressed.

