import sys
import matplotlib.pyplot as plt

scores = []
# exploration_rates = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        scores.append(float(line.split(',')[1]))
        # exploration_rates.append(float(line.split(',')[1])*100)

# k = 100
# sliding_avg = []
# avg = sum(scores[0:k])/k
# for i in range(len(scores) - k):
#     sliding_avg.append(avg)
#     avg = avg + scores[i+k]/k - scores[i]/k
plt.plot(scores, label="avg game score")
# plt.plot(sliding_avg, label=f"sliding over {k} average score")
# plt.plot(exploration_rates, label="exploration rate (%)")
plt.legend()
plt.show()