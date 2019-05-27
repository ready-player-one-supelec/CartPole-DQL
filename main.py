import sys
import random
import time
import gym
import numpy as np
from perceptron import MultiLayerPerceptron as MLP

def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm


def run(numero, run_numero, games, games_before_test, test_games, exploration_rate, render, treshold, log):
    ## INIT
    time.perf_counter()
    env = gym.make('CartPole-v1')
    file = open(f'cartpole_{numero}.{run_numero}.txt', 'w')
    file.close()
    # long_mem = []

    mlp = MLP([4, 16, 16, 2])

    # TRAINING
    for i in range(games):
        state = env.reset()
        done = False
        score = 0
        short_mem = []
        # Play a game
        while not done:
            if random.random() < exploration_rate:
                action = random.randint(0, 1)
            else:
                actions = mlp.frontprop(normalize(np.array(state)))
                action = np.argmax(actions)
                # action = np.argmin(actions)
            short_mem.append((state, action))
            state, reward, done, _ = env.step(action)
            if render: env.render()
            score += 1
        # If win then learn short mem
        if score == 500:
            for state, action in short_mem:
                mlp.backprop(normalize(np.array(state)), np.array([1, 0] if action == 0 else [0, 1]))
                mlp.fit()
                # long_mem.append((state, action))
        # If loose, don't
        else:
            pop = True
            while pop:
                state, action = short_mem.pop()
                actions = mlp.frontprop(normalize(np.array(state)))
                # Action chosen is bad
                if action == 0:
                    expected = np.array([0, actions[1]])
                else:
                    expected = np.array([actions[0], 0])
                # Learn corrected action
                mlp.backprop(normalize(np.array(state)), expected)
                mlp.fit()
                # Check treshold, if all under propagation, repop and relearn
                if len(short_mem) == 0 or (mlp.frontprop(normalize(np.array(state)))>0.1).any():
                    pop = False
            # long_mem.append((state, action))
        # Random learning in memory
        # for k in range(mem_size):
        #     state, action = random.choice(long_mem)
        #     mlp.backprop(normalize(np.array(state)), np.array([1, 0] if action == 1 else [0, 1]))
        #     mlp.fit()
        # Test
        if i % games_before_test == 0:
            avg_score = 0
            for k in range(test_games):
                state = env.reset()
                done = False
                score = 0
                # Play a game
                while not done:
                    actions = mlp.frontprop(normalize(np.array(state)))
                    action = np.argmax(actions)
                    # action = np.argmin(actions)
                    state, reward, done, _ = env.step(action)
                    if render: env.render()
                    score += 1
                avg_score += score
            avg_score /= test_games
            # Logging
            file = open(f'cartpole_{numero}.txt', 'a')
            print(time.perf_counter(), avg_score, sep=',', file=file)
            if log: print(f"Game {i}: score {avg_score} avg over {test_games} games, exploration {exploration_rate}")
            file.close()
        # Decrease exploration_rate
        exploration_rate = exploration_rate - 1/games

    env.close()
