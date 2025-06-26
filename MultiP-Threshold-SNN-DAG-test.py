import numpy as np
from SNN_neat import DAG, plot_dag
from tqdm import tqdm
import plotly.graph_objects as go
import json
from multiprocessing import Process, Manager
from time import time

NETWORK_WIDTH = 100
MAX_THREADS = 10
EVALUATIONS = 100

class EvaluationProcess(Process):
    def __init__(self, threshold, percentages, count, return_dict):
        super().__init__()
        self.threshold = threshold
        self.percentages = percentages
        self.return_dict = return_dict
        self.test_count = count

    def run(self):
        dag = DAG(NETWORK_WIDTH, NETWORK_WIDTH)

        # create DAG
        for i, node_id in enumerate(dag.input_nodes):
            output_node = dag.output_nodes[i]
            dag.add_connection(node_id, output_node)
            _, new_node = dag.add_node(node_id, output_node)
            _, new_node2 = dag.add_node(node_id, new_node)
            _, new_node3 = dag.add_node(new_node, output_node)

        dag.processing_order = dag.get_processing_order()
        for i, layer in enumerate(dag.processing_order[:-1]):
            for node1 in layer:
                for node2 in dag.processing_order[i+1]:
                    dag.add_connection(node1, node2)

        # evaluate
        dag.processing_order = dag.get_processing_order()
        evaluation = {'%': [], 'mean':[], 'max':[], 'min': [], 'std': []}  # input activity, mean activity
        for percentage in self.percentages:
            # set threshold
            for layer in dag.processing_order:
                for nodeId in layer:
                    node = dag.get_node(nodeId)
                    node.threshold = self.threshold

            layer_avg_activations = [[] for _ in range(len(dag.get_processing_order()))]

            # iterate multiple times
            for _ in range(self.test_count):
                dag.reset()
                inputs = [int(np.random.rand() < percentage) for _ in range(NETWORK_WIDTH)]

                for layer in dag.processing_order:
                    for nodeId in layer:
                        node = dag.get_node(nodeId)
                        new_weights = np.random.rand(len(node.weights)) * 2 - 1
                        node.bias = np.random.rand() * 2 - 1
                        node.weights = new_weights / np.linalg.norm(list(new_weights)+[node.bias])
                        node.bias = node.bias / np.linalg.norm(list(new_weights)+[node.bias])

                dag.process_once(inputs)

                # get activation
                for i, layer in enumerate(dag.get_processing_order()):
                    avg_activation_probability = 0
                    for node_id in layer:
                        node = dag.get_node(node_id)
                        avg_activation_probability += node.spike

                    avg_activation_probability = avg_activation_probability / len(layer)

                    layer_avg_activations[i].append(avg_activation_probability)

            evaluation['%'].append(percentage)
            evaluation['mean'].append(np.mean(layer_avg_activations[1:-1]))
            evaluation['std'].append(np.std(layer_avg_activations[1:-1]))
            evaluation['max'].append(np.max(layer_avg_activations[1:-1]))
            evaluation['min'].append(np.min(layer_avg_activations[1:-1]))

        self.return_dict[self.threshold] = evaluation

# yields the threshold that have to be evaluated
def range_iterator(thresholds):
    for threshold in thresholds:
        yield threshold

# evaluates the thresholds in the given range
def evaluate_threshold(max_count, thresholds, percentages, evaluations):
    threshold_iterator = range_iterator(thresholds)
    manager = Manager()
    results = manager.dict()

    iterations = len(thresholds) // max_count
    if iterations == 0:
        iterations = 1

    for i in range(iterations):
        print(f"------------------ Batch: {i+1}/{iterations} --------------------")
        # create max amount of processes
        processes = []
        for _ in range(max_count):
            try:
                threshold = next(threshold_iterator)
                process = EvaluationProcess(threshold, percentages, evaluations, results)
                process.start()
                processes.append(process)
            except StopIteration:
                break
        # wait until all processes are done
        for process in tqdm(processes, desc="Processes Done"):
            process.join()

    return dict(results)

if __name__ == '__main__':
    percentages = np.arange(0, 1.1, 0.01)
    thresholds_range = np.arange(-1, 0.7, 0.01) 

    start_time = time()
    thresholds = evaluate_threshold(MAX_THREADS, thresholds_range, percentages, EVALUATIONS)
    print(f"Time: {time() - start_time}\n\n")


    # plot all the tested thresholds, by different lines in the xs vs. ys plot with plotly
    # x values are input activity, y values are mean activity of the second layer
    fig = go.Figure()
    for threshold, evaluation in thresholds.items():
        xs = evaluation['%']
        fig.add_trace(go.Scatter(x=xs, y=np.array(evaluation['min']), mode='lines+markers', name=f"T: {threshold}"))
        #fig.add_trace(go.Scatter(x=xs, y=np.array(evaluation['max']), mode='lines+markers', name=f"T: {threshold}"))
        #fig.add_trace(go.Scatter(x=xs, y=np.array(evaluation['mean']), mode='lines', name=f"T: {threshold}"))

    fig.add_trace(go.Scatter(x=xs, y=xs, mode='lines', name="y=x"))
    fig.update_xaxes(title="Input activity")
    fig.update_yaxes(title="Mean activity")
    fig.update_layout(title=f"T: {threshold}")
    fig.show()

    # extract for all input activities all the threshold where the mean activity is the closest to the input activity
    print(percentages.shape)
    closest_thresholds = [[], []]
    for input_percentage in percentages:
        closest_mean_threshold = None
        closest_distance = float("inf")
        for threshold, evaluation in thresholds.items():
            xs = evaluation['%']
            ys = evaluation['mean']
            distance = abs(input_percentage - ys[xs.index(input_percentage)])
            if distance < closest_distance:
                closest_distance = distance
                closest_mean_threshold = threshold
        closest_thresholds[0].append(closest_mean_threshold)
        closest_thresholds[1].append(closest_distance)

    # plot the closest thresholds for each input activity
    fig = go.Figure()
    fig.update_layout(title=f"Closest Thresholds")
    fig.add_trace(go.Scatter(x=percentages, y=closest_thresholds[0], mode='lines+markers'))
    fig.add_trace(go.Scatter(x=percentages, y=closest_thresholds[1], mode='lines+markers'))
    fig.show()

    # save the thresholds to a json file
    with open("thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=4)

    closest_thresholds = {}
    for input_percentage in percentages:
        closest_thresholds[input_percentage] = {"threshold": None, "distance": float("inf"), "min": None, "max": None}
        closest_mean_threshold = None
        closest_min_threshold = None
        closest_max_threshold = None
        closest_distance = float("inf")

        for threshold, evaluation in thresholds.items():
            xs = evaluation['%']
            idx = xs.index(input_percentage)
            distance = abs(input_percentage - evaluation['mean'][idx])
            if distance < closest_distance:
                closest_distance = distance
                closest_mean_threshold = threshold
                closest_min_threshold = evaluation['min'][idx]
                closest_max_threshold = evaluation['max'][idx]

        closest_thresholds[input_percentage]["mean"] = closest_mean_threshold
        closest_thresholds[input_percentage]["distance"] = closest_distance
        closest_thresholds[input_percentage]["min"] = closest_min_threshold
        closest_thresholds[input_percentage]["max"] = closest_max_threshold

    # save the closest thresholds to a json file
    with open("closest_thresholds.json", "w") as f:
        json.dump(closest_thresholds, f, indent=4)