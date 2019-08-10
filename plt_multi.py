import sys
import matplotlib.pyplot as plt
import math
from conf import runs, games, games_before_test

numero = int(sys.argv[1])

scores = [0] * (games//games_before_test)
for k in range(runs):
    i = 0
    file = open(f'data/cartpole_{numero}.{k}.txt', 'r')
    for line in file:
        scores[i] += float(line.split(',')[1])
        i += 1
    file.close()

error = []
for i in range(games//games_before_test):
    scores[i] /= runs
    error.append(1.96 * scores[i] / math.sqrt(runs))

# plt.errorbar(x=list(range(games//games_before_test)), y=scores, yerr=error, label=f"avg game score over {runs} runs")
plt.plot(scores)
plt.show()