from main import run
import sys
import random
import multiprocessing
from conf import *



numero = 0
if len(sys.argv) < 2 or not sys.argv[1].isdigit():
    numero = random.randint(0, 100)
else:
    numero = int(sys.argv[1])

def f(k):
    return run(numero, k, games, games_before_test, test_games, exploration_rate, render, treshold, log=True, gamma=gamma)

p = multiprocessing.Pool()
p.map(f, range(runs))
