from main import run
import sys
import random
import multiprocessing


## CONFIG
games = 1000
games_before_test = 10
test_games = 20
exploration_rate = 1
render = False
# mem_size = 64
treshold = 0.1
runs = 100

numero = 0
if len(sys.argv) < 2 or not sys.argv[1].isdigit():
    numero = random.randint(0, 100)
else:
    numero = int(sys.argv[1])

def f(k):
    return run(numero, k, games, games_before_test, test_games, exploration_rate, render, treshold, log=True)

p = multiprocessing.Pool(runs)
p.map(f, range(runs))
