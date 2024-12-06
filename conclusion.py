import json
import matplotlib.pyplot as plt
import numpy as np
import os
# Load the data from the JSON file
data = {}
for algo in ('nb', 'knn', 'svc', 'sgd'):
    with open(f'{algo}-metrics.json') as f:
        algo_metrics = json.load(f)
        algo_metrics = {"weighted avg precision" : algo_metrics["weighted avg"]["precision"], 
               "weighted avg recall" : algo_metrics["weighted avg"]["recall"], 
               "weighted avg f1-score" : algo_metrics["weighted avg"]["f1-score"], 
               "training time" : algo_metrics["Training time"], 
               "prediction time" : algo_metrics["Prediction time"], 
               "model_size (MB)" : os.stat(f'{algo}.joblib').st_size / (1024 * 1024)}
        data[algo] = algo_metrics
# create 5 bar graphs, where algorithms are compared to each other using the accuracy metric; the weighted average metric; etc.
# the x-axis should be the algorithms, the y-axis should be the metric values
# the bars should be side by side, with a legend indicating which metric is which

# create a list of algorithms
algorithms = list(data.keys())
# create a list of metrics
metrics = list(data[algorithms[0]].keys())
# create a list of metric values for each algorithm
metric_values = [[data[algo][metric] for algo in algorithms] for metric in metrics]
print(metric_values[4])
# create a list of positions for the bars
positions = np.arange(len(algorithms))
# create a figure and axis

# create a bar graph for each metric
for i, metric in enumerate(metrics):
    fig, ax = plt.subplots()
    ax.bar(positions + i * 0.1, metric_values[i], width=0.1, color='r', label=metric)
    ax.set_xticks(positions + 0.2)
    ax.set_xticklabels(algorithms)
    ax.set_xlabel('Алгоритм')
    ax.set_ylabel('Значение')
    ax.set_title('Сравнение алгоритмов')
    ax.legend()
    fig.autofmt_xdate()
    plt.savefig(f'{metric}_comparison_plot.png')
    #clear plot
    plt.clf()
