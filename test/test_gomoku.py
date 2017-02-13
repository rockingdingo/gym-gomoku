import gym
import gym_gomoku
env = gym.make('Gomoku9x9-v0')

# example 1: take an action
env.reset()
env.render()
env.step(15) # place a single stone, black color first

# example 2: playing with beginner policy, with basic strike and defend 'opponent'
env.reset()
for _ in range(20):
    action = env.action_space.sample() # sample without replacement
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        print ("Game is Over")
        break
