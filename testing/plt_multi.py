import sys
import matplotlib.pyplot as plt
import math
from conf import runs, games, games_before_test

numero = int(sys.argv[1])

scores = [0] * (games//games_before_test)
for k in range(runs):
    scores = []
    file = open(f'data/cartpole_{numero}.{k}.txt', 'r')
    for line in file:
        scores.append(float(line.split(',')[1]))
    plt.plot(scores)
    file.close()


plt.show()
