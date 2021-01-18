# Reinforcement-Learning-QLearning
This repo contains simple q-learning implementation for taxi-v3 environment of Open AI gym.

Read the tutorial here https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

![alt-text](https://github.com/akshitasawhney3008/Reinforcement-Learning-QLearning/blob/main/taxi_v3.gif)

## Rewards and Penalties:
  * High positive reward for a successful dropoff
  * Penalized if it tries to drop off a passenger in wrong locations
  * slight negative reward for not making it to the destination after every time-step.
  
## State Space: 
The State Space is the set of all possible situations our taxi could inhabit.
Taxi environment has 5×5×5×4=500 total possible states.

## Action Space:
Six possible actions:

        *south
        *north
        *east
        *west
        *pickup
        *dropoff


### The workspace contains three files:
agent.py: Developed reinforcement learning agent here.
monitor.py: The interact function tests how well your agent learns from interaction with the environment.
main.py: Run this file in the terminal to check the performance of your agent.
