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


## Installing required libraries.
pip3 install -r Requirements.txt

## Running the files
Main file: Environment_setting.py is where we set the environment we want the agent to be trained in. Also here we decide the parameters that we want to set before training the agent.

Set train_dqn = 0 to train simple q learning.

## Other files in workspace
Train_MyRLAgent.py is called to start training the agent

QLearning_Agent.py is where the Agent is created

## Plots
Plots of rewards, averaged rewards , steps and epsilo verses the number of episoded are created to show whether the agent is correctly getting trained.
