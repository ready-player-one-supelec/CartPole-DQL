import gym
import time

env = gym.make('CartPole-v1')
done = False
state = env.reset()
k = 0
while not done:
    k += 1
    print(state)
    time.sleep(0.05)
    if state[3] < 0:
        state, reward, done, info = env.step(0)
    else:
        state, reward, done, info = env.step(1)
    env.render()
print(k)
env.close()
