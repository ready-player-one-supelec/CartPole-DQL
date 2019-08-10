import sys
import matplotlib.pyplot as plt
import math
from conf import runs, games, games_before_test

numero = int(sys.argv[1])

games = 100000

scores = [0] * (games//games_before_test)
sigmas = [0] * len(scores)

for k in range(runs):
    file = open(f'data/cartpole_{numero}.{k}.txt', 'r')
    for j in range(len(scores)):
        line = file.readline()
        score = float(line.split(',')[1])
        scores[j] += score
        sigmas[j] += score**2
    file.close()


error = []
for i in range(len(scores)):
    scores[i] /= runs
    sigmas[i] = math.sqrt(sigmas[i] / runs - scores[i]**2)
    if i%10 == 0:
    	error.append(1.96 * sigmas[i] / math.sqrt(runs))
    else:
        error.append(0)

print("30 000", "score", scores[30000//games_before_test], "sigma", sigmas[30000//games_before_test])
plt.errorbar(x=list(range(games//games_before_test)), y=scores, yerr=error, label=f"avg game score over {runs} runs")
plt.ylabel("Score moyen sur 100 runs")
plt.xlabel("Nombre de parties d'entrainement")
plt.show()
