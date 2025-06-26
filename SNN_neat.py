import numpy as np
#from uuid import uuid1
import plotly.graph_objects as go
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from random import randint, random as rand, choice
from time import time
from multiprocessing import Manager, Pool
import multiprocessing as mp
import copy
from math import isnan
import random

def dummy_uuid_generator():
    i = 0
    while True:
        yield str(i)
        i += 1

dummy_gen = dummy_uuid_generator()

def uuid1():
    return next(dummy_gen)

VALUE_CAP = 5000
TIME_RANGE = 5

class Sigmoid:
    def __call__(self, x):
        # Cap the input to avoid overflow in exp(-x)
        x = np.clip(x, -VALUE_CAP, VALUE_CAP)
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        sigmoid_x = self(x)
        return sigmoid_x * (1 - sigmoid_x)

class ReLU:
    def __call__(self, x):
        # ReLU itself is stable, but we can cap the inputs to prevent extreme outputs
        return np.where(x > 0, np.clip(x, 0, 500), 0)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def __call__(self, x):
        return np.where(x > 0, np.clip(x, 0, 500), self.alpha * np.clip(x, -VALUE_CAP, VALUE_CAP))
    
    def derivative(self, x):
        return np.where(x > 0, 1, self.alpha)

class Elu:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, x):
        # Cap input to prevent exp(x) from producing very large values
        return np.where(x > 0, np.clip(x, 0, 500), self.alpha * (np.exp(np.clip(x, -VALUE_CAP, VALUE_CAP)) - 1))
    
    def derivative(self, x):
        elu_x = self(x)
        return np.where(x > 0, 1, elu_x + self.alpha)

class Tanh:
    def __call__(self, x):
        # Tanh is naturally bounded between -1 and 1, but we can cap the input to avoid overflow
        x = np.clip(x, -VALUE_CAP, VALUE_CAP)
        return np.tanh(x)
    
    def derivative(self, x):
        tanh_x = self(x)
        return 1 - tanh_x**2

class Linear:
    def __call__(self, x):
        # Linear activation is naturally unbounded, so you might consider capping outputs if needed
        return np.clip(x, -VALUE_CAP, VALUE_CAP)
    
    def derivative(self, x):
        return np.ones_like(x)

class MSE:
    def __call__(self, y, t):
        return np.sum((y - t)**2) / len(y)
    
    def derivative(self, y, t):
        return -(t - y)

class CrossEntropy:
    def __call__(self, y, t):
        # Clip y to avoid log(0) and log(1)
        epsilon = 1e-10
        y = np.clip(y, epsilon, 1 - epsilon)
        return -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y))
    
    def derivative(self, y, t):
        # Clip y to avoid division by zero
        epsilon = 1e-10
        y = np.clip(y, epsilon, 1 - epsilon)
        return (y - t) / (y * (1 - y))

class Softmax:
    def __call__(self, x):
        # Subtract the max value from x to prevent overflow in exp
        x = np.clip(x, -VALUE_CAP, VALUE_CAP)
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    
    def derivative(self, x):
        # Softmax derivative is complex, but we keep the output stable
        softmax_x = self(x)
        return softmax_x * (1 - softmax_x)

import numpy as np

class Node:
    def __init__(self, id, initial_threshold=0.7):
        self.id = id
        self.input_cons = []
        self.output_cons = []
        self.weights = np.array([])
        self.bias = np.random.random() * 2 - 1
        self.threshold = initial_threshold
        self.int_val = 0
        self.spike = 0
        self.activity = []
        self.last_spike_time = -np.inf  # Use a large negative value to indicate no spike has occurred
        self.time = 0  # This should be updated externally, representing the current time step

    def add_input_con(self, id):
        self.input_cons.append(id)
        weights = list(self.weights)
        weights.append(np.random.random() * 2 - 1)  # Small random values for initial weights
        self.weights = np.array(weights)
        #self.weights = weights / np.linalg.norm(weights)

    def add_output_con(self, id):
        self.output_cons.append(id)

    def remove_input_con(self, id):
        idx = self.input_cons.index(id)
        w = list(self.weights)
        w.pop(idx)
        self.weights = np.array(w)
        self.input_cons.remove(id)

        #self.weights /= np.linalg.norm(self.weights)

    def remove_output_con(self, id):
        self.output_cons.remove(id)

    def set_internal_value(self, val):
        self.int_val = val

    def process(self, inputs=[]):
        if len(inputs):
            self.int_val = np.dot(inputs, self.weights)
            self.int_val += self.bias

        self.spike = int(self.int_val >= self.threshold)
        if self.spike:
            self.last_spike_time = self.time

        self.activity.append(self.spike)  # Track activity

        """ if len(self.activity) > TIME_RANGE:
            self.activity = self.activity[1:] """

        #print(f"Update {self.id} with inputs: {inputs}, weights: {self.weights}, bias: {self.bias}, threshhold: {self.threshold}, result: {self.int_val} | {self.spike}")

    def calculate_sigma(self, sigma=1):
        if len(self.weights) == 0:
            return self.weights, self.bias
        
        pm = np.where(self.weights > 0, self.weights, 0)
        nm = np.where(self.weights < 0, self.weights, 0)

        ps = (np.sum(pm) + (self.bias > 0) * self.bias)
        ns = - (np.sum(nm) + (self.bias < 0) * self.bias)

        if ps == 0:
            # flip ranfom positive weight
            i = np.random.randint(0, len(self.weights))
            self.weights[i] = abs(self.weights[i])
            pm = np.where(self.weights > 0, self.weights, 0)
            ps = np.sum(pm)

        r = ((sigma * self.threshold) - (ps - ns)) / len(self.weights + (self.bias != 0))
        return self.weights + r, (self.bias + r) * (self.bias != 0)

    def update_weights(self, inputs, global_error):
        for i, in_node_id in enumerate(self.input_cons):
            weight_change = 0
            # strength
            if inputs[i] and not self.spike and (global_error > 0):
                weight_change = 0.1 * global_error
            # weaken
            elif inputs[i] and self.spike and (global_error < 0):
                weight_change = 0.1 * global_error
            
            if not inputs[i] and not self.spike and (global_error > 0):
                self.bias += 0.01 * global_error

            if not inputs[i] and self.spike and (global_error < 0):
                self.bias += 0.01 * global_error
                
            if global_error != 0:
                #print(f"{self.id} -> {in_node_id} {inputs[i]}:\n\tstrength: {strengthen} | weaken: {weaken}\n\tglobal error: {(global_error)}\n\td_weight: {weight_change}\n\tweight befor: {self.weights[i]}\n\tweight after: {self.weights[i]+weight_change}")
                pass

            self.weights[i] += weight_change
            # Ensure the weights stay within a reasonable range
            #self.weights, self.bias = self.calculate_sigma()

    def get_activity(self):
        return np.mean(self.activity)

    def update_threshold(self, input_activities):
        if len(input_activities):
            #print(f"update Threshold {self.id}: {input_activities}: {self.threshold} -> {self.calculate_threshold(np.mean(input_activities))}")
            avg_activity = np.mean(input_activities)
            #self.threshold = self.calculate_threshold(avg_activity)

    def calculate_threshold(self, activation_prob):
        return 1.362932 * np.power(activation_prob, 0.070175) + -1.299556

    def reset(self):
        self.int_val = 0
        self.spike = 0
        self.activity = []  # Reset activity after each epoch or batch
        self.last_spike_time = 0  # Reset spike timing
        self.time = 0  # Reset time to start a new epoch

    def __str__(self) -> str:
        return f"SpikingNode({self.id}: bias: {self.bias}, threshold: {self.threshold}, in: {[str(n) for n in self.input_cons]} out: {[str(n) for n in self.output_cons]}, weights: {self.weights})"

class Layer:
    def __init__(self, nodes):
        self.nodes = nodes  # List of node IDs that belong to this layer

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx]
    
    def __contains__(self, id):
        return id in self.nodes

    def __str__(self):
        return f"Layer({', '.join(self.nodes)})"

    def process(self, dag, processed_dict):
        """
        Process the nodes in the layer, calculate outputs, and apply STDP.
        """
        for node_id in self.nodes:
            node = dag.get_node(node_id)
            input_values = [processed_dict[in_node_id] for in_node_id in node.input_cons]
            input_values = np.array(input_values)
            node.process(input_values.T)
            processed_dict[node_id] = node.spike

        return processed_dict

    def update_thresholds(self, dag):
        """
        Update the thresholds of nodes in this layer based on input activities.
        """
        for node_id in self.nodes:
            node = dag.get_node(node_id)
            input_activities = [dag.get_node(in_node_id).get_activity() for in_node_id in node.input_cons]
            node.update_threshold(input_activities)

class MLayer:
    def __init__(self, nodes):
        self.nodes = nodes  # List of node IDs that belong to this M-Layer
        self.previous_outputs = {node_id: 0 for node_id in nodes}  # Initialize previous outputs to zero

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx]

    def __str__(self):
        return f"MLayer({', '.join(self.nodes)})"

    def process(self, dag, processed_dict):
        """
        Process the M-Layer nodes in a random order, using previous iteration values for inputs that have not been processed yet.
        """
        # Shuffle the order of node evaluation
        random.shuffle(self.nodes)

        # Process each node in the M-Layer
        for node_id in self.nodes:
            node = dag.get_node(node_id)
            input_values = []

            for in_node_id in node.input_cons:
                if in_node_id in processed_dict:
                    input_values.append(processed_dict[in_node_id])
                else:
                    input_values.append(self.previous_outputs[in_node_id])  # Use the last iteration's value

            input_values = np.array(input_values)
            node.process(input_values.T)
            processed_dict[node_id] = node.out_val

        # Update the previous outputs after processing the layer
        for node_id in self.nodes:
            self.previous_outputs[node_id] = processed_dict[node_id]

        return processed_dict

    def update_thresholds(self, dag):
        """
        Update the thresholds of nodes in this M-Layer based on input activities.
        """
        for node_id in self.nodes:
            node = dag.get_node(node_id)
            input_activities = [dag.get_node(in_node_id).get_activity() for in_node_id in node.input_cons]
            node.update_threshold(input_activities)

class DAG: 
    def __init__(self, nr_inputs, nr_outputs, id='0', do_setup=True, fully_connect=False, 
                error_function=MSE(), 
                ):
        self.id = id
        self.nodes = []
        self.input_nodes = []
        self.output_nodes = []
        self.processing_order = []
        self.nr_inputs = nr_inputs
        self.nr_outputs = nr_outputs
        self.error_function = error_function

        if do_setup:
            for i in range(nr_inputs):
                self.nodes.append(Node(uuid1()))
                self.input_nodes.append(self.nodes[-1].id)
            
            for i in range(nr_outputs):
                self.nodes.append(Node(uuid1()))
                self.output_nodes.append(self.nodes[-1].id)

            if fully_connect:
                for node in self.input_nodes:
                    for other_node in self.output_nodes:
                        if node == other_node:
                            continue
                        self.add_connection(node, other_node)

            self.processing_order = self.get_processing_order()
        
    def get_node(self, id) -> Node:
        for node in self.nodes:
            if node.id == id:
                return node
        return None
    
    def get_nodes(self):
        return self.nodes
    
    def get_connections(self) -> list:
        connections = []
        for node in self.nodes:
            for con in node.output_cons:
                connections.append((node.id, con))
        return connections

    def does_create_cycle(self, nod1_id, nod2_id) -> bool:
        """
        Check if adding a connection from nod1_id to nod2_id would create a cycle.
        Bidirectional connections are allowed, so we ignore them in the cycle check.
        """
        connections = self.get_connections()
        new_edge = (nod1_id, nod2_id)

        # If the connection already exists in the opposite direction, it's a bidirectional connection.
        if self.is_connected(nod2_id, nod1_id):
            return False  # This would be a new bidirectional connection, not a cycle

        # Convert connections to an adjacency list
        graph = defaultdict(list)
        for start, end in connections:
            # Skip bidirectional connections to prevent them from being considered in cycle detection
            if self.is_connected(end, start):
                continue
            graph[start].append(end)
        
        def dfs(node, visited, stack):
            visited.add(node)
            stack.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor, visited, stack):
                        return True
                elif neighbor in stack:
                    return True
                    
            stack.remove(node)
            return False
        
        # Add the new edge to the graph
        graph[new_edge[0]].append(new_edge[1])
        
        visited = set()
        stack = set()
        
        # Perform DFS to check for cycles starting from the new edge's source
        if dfs(new_edge[0], visited, stack):
            return True
        return False

    def get_legal_connections(self) -> list:
        '''
        Get all possible connections that can be made without corrupting the DAG
        Not allowed are:
        - connections that have input nodes as outputs
        - connections that have output nodes as inputs
        - connections that are already present
        - connections that go back in the processing order
        '''
        legal_connections = []
        for n1 in self.nodes:
            for n2 in self.nodes:
                # if n1 is an output node
                if n1.id in self.output_nodes:
                    continue
                # if n2 isan input node
                if n2.id in self.input_nodes:
                    continue
                # if nodes are the same
                if n1.id == n2.id:
                    continue
                # if connection is already present
                if self.is_connected(n1.id, n2.id):
                    continue
                # if would create a cycle
                if self.does_create_cycle(n1.id, n2.id):
                    continue
                
                legal_connections.append((n1.id, n2.id))
        return legal_connections

    def is_connected(self, nod1_id, nod2_id) -> bool:
        node1 = self.get_node(nod1_id)
        node2 = self.get_node(nod2_id)
        return node2.id in node1.output_cons
    
    def add_node(self, node1_id, node2_id, activation_function=None):
        if not self.is_connected(node1_id, node2_id):
            return False
        
        new_node = Node(uuid1())
        self.nodes.append(new_node)
        self.remove_connection(node1_id, node2_id)
        self.add_connection(node1_id, new_node.id)
        self.add_connection(new_node.id, node2_id)
        return True, new_node.id

    def add_connection(self, nod1_id , nod2_id) -> bool:
        if self.is_connected(nod1_id, nod2_id):
            return False
        node1 = self.get_node(nod1_id)
        node2 = self.get_node(nod2_id)
        node1.add_output_con(node2.id)
        node2.add_input_con(node1.id)
        return True

    def remove_connection(self, nod1_id, nod2_id):
        node1 = self.get_node(nod1_id)
        node2 = self.get_node(nod2_id)
        node1.remove_output_con(node2.id)
        node2.remove_input_con(node1.id)

    def remove_node(self, node_id):
        node = self.get_node(node_id)
        while len(node.input_cons):
            con  = node.input_cons[0]
            self.remove_connection(con, node_id)
        while len(node.output_cons):
            con  = node.output_cons[0]
            self.remove_connection(node_id, con)
        self.nodes.remove(node)

    def kill_iteration(self) -> bool:
        for node in self.nodes:
            for node in self.nodes:
                if node.id in self.input_nodes or node.id in self.output_nodes:
                    continue
                if len(node.input_cons) == 0 or len(node.output_cons) == 0:
                    self.remove_node(node.id)
                    return True
        return False

    def kill_floaters(self):
        found_one = True
        while found_one:
            found_one = self.kill_iteration()

    def check_for_circular(self):
        '''
        Check if the graph has circular connections
        do a depth first search from each node and see if it can reach any of the already visited nodes
        '''
        visited = set()
        for node in self.nodes:
            stack = [node]
            while stack:
                n = stack.pop()
                if n.id in visited:
                    return True
                visited.add(n.id)
                for con in n.output_cons:
                    stack.append(self.get_node(con))
        return False

    def get_nodes_as_dict(self):
        nodes = {}
        for node in self.nodes:
            nodes[node.id] = {"input_cons": node.input_cons.copy(), "output_cons": node.output_cons.copy()}
        return nodes

    def identify_m_layers(self):
        """Identify all M-Layers using strongly connected components (SCC)."""
        G = nx.DiGraph()

        # Create a directed graph of the current DAG
        for node in self.get_nodes():
            G.add_node(node.id)

        for con in self.get_connections():
            G.add_edge(con[0], con[1])

        # Identify strongly connected components
        sccs = list(nx.strongly_connected_components(G))

        m_layers = []
        other_nodes = set(node.id for node in self.get_nodes())
        m_layer_nodes = []

        # Create M-Layers from SCCs that have more than one node
        for scc in sccs:
            if len(scc) > 1:
                m_layer = MLayer(list(scc))
                m_layers.append(m_layer)
                other_nodes -= scc  # Remove these nodes from the set of other nodes
                m_layer_nodes += list(scc)

        return m_layers, list(other_nodes), m_layer_nodes

    def get_processing_order(self) -> list:
        # Get all M-Layers
        m_layers, remaining_nodes, m_layer_nodes = self.identify_m_layers()

        processing_order = []

        # Topological sorting of the DAG with virtual nodes
        nodes = self.get_nodes_as_dict()

        # create list with sets of inputs for m-layers
        v_nodes = {}
        for i, m_layer in enumerate(m_layers):
            inputs = []
            for node_id in m_layer:
                inputs += nodes[node_id]["input_cons"]

            inputs_set = []
            for input in set(inputs):
                if input not in m_layer:
                    inputs_set.append(input)

            inputs_set = set(inputs_set)

            v_nodes[i] = inputs_set

        while nodes:
            new_layer = []
            for node_id in list(nodes.keys()):
                # check if m layer node
                if node_id in m_layer_nodes:
                    continue
                # Check if all input nodes are processed
                if all((input_id not in nodes) for input_id in nodes[node_id]["input_cons"]):
                    new_layer.append(node_id)

            # if a layer has been found
            if len(new_layer):
                processing_order.append(Layer(new_layer))
                for node_id in new_layer:
                    del nodes[node_id]
            # if no layer found, check if M-Layer is done
            else:
                found_m_layer = -1
                for key, input_set in v_nodes.items():
                    clear_flag = True
                    for input in input_set:
                        if input in nodes:
                            clear_flag = False
                            break
                    
                    if clear_flag:
                        found_m_layer = key
                        break
                
                if found_m_layer >= 0:
                    processing_order.append(m_layers[found_m_layer])
                    del v_nodes[found_m_layer] # remove from set

                    # remove nodes from layer from dict
                    for node_id in m_layers[found_m_layer]:
                        del nodes[node_id]

        return processing_order

    def process(self, inputs, iterations=5, return_spiketrains=False):
        inputs = np.array(inputs)

        # Spiketrain collection
        spiketrains = [[] for _ in self.output_nodes]

        # Iteratively process the network
        for _ in range(iterations):
            processed_dict = {}
            # Set all inputs to the input nodes
            for i, node_id in enumerate(self.input_nodes):
                node = self.get_node(node_id)
                node_inputs = inputs[:, i] if inputs.ndim > 1 else np.array([inputs[i]])
                node.set_internal_value(node_inputs)

            # iterate over layers
            for layer in self.processing_order:
                layer.process(self, processed_dict)  # Delegate processing to the Layer or MLayer

            # Update thresholds after processing all nodes in this iteration
            for layer in self.processing_order:
                layer.update_thresholds(self)

            # Track spiketrains
            for node_id in self.output_nodes:
                spiketrains[self.output_nodes.index(node_id)].append(self.get_node(node_id).spike)

        if return_spiketrains:
            return np.array(spiketrains)
        else:
            # Return list of spikes in the order of output_nodes
            return np.mean(spiketrains, axis=1)

    def process_once(self, inputs, return_spiketrains=False):
        inputs = np.array(inputs)
        processed_nodes = {}

        # Spiketrain collection
        spiketrains = [[] for _ in self.output_nodes]

        # Set all inputs to the input nodes
        for i, node_id in enumerate(self.input_nodes):
            node = self.get_node(node_id)
            node_inputs = inputs[:, i] if inputs.ndim > 1 else np.array([inputs[i]])
            node.set_internal_value(node_inputs)

        # iterate over layers
        for layer in self.processing_order:
            processed_nodes = layer.process(self, processed_nodes)  # Delegate processing to the Layer or MLayer

        # Update thresholds after processing all nodes in this iteration
        """ for layer in self.processing_order:
            layer.update_thresholds(self) """

        # Track spiketrains
        for node_id in self.output_nodes:
            spiketrains[self.output_nodes.index(node_id)].append(self.get_node(node_id).spike)

        if return_spiketrains:
            return np.array(spiketrains)
        else:
            # Return list of spikes in the order of output_nodes
            return np.mean(spiketrains, axis=1)

    def cost(self, outputs, targets):
        return self.error_function(np.array(outputs), np.array(targets))

    def train(self, inputs, targets, epochs=100, lr=0.01, verbose=False):
        """
        Train the DAG network using individual input patterns, evaluating one pattern at a time.
        """
        if verbose:
            print(f"Start Training DAG {self.id} with {epochs} epochs")

        self.processing_order = self.get_processing_order()
        losses = []

        # Iterate over each epoch
        for epoch in tqdm(range(epochs), disable=not verbose):
            epoch_loss = 0

            # Iterate over each input-target pair
            for input_pattern, target in zip(inputs, targets):
                # Reset the network before processing each new input pattern
                self.reset()

                # Process the current input pattern (single sample)
                output = self.process(input_pattern, iterations=10, return_spiketrains=False)

                # Compute the global error for this pattern
                loss = self.cost(output, target)
                epoch_loss += loss
                error = np.sum(np.array(target)-np.array(output))

                #print(f"Inputs: {input_pattern} -> {output} = {target} ")

                # Apply STDP with global error modulation
                for layer in self.processing_order:
                    for node_id in layer:
                        node = self.get_node(node_id)
                        input_values = [self.get_node(in_node_id).spike for in_node_id in node.input_cons]
                        node.update_weights(input_values, error)

            # Store the average loss over all patterns in this epoch
            losses.append(epoch_loss / len(inputs))

        return losses


    @classmethod
    def from_other(cls, other):
        this = cls(other.nr_inputs, other.nr_outputs, do_setup=False)
        # copy the class variables
        this.id = other.id
        this.error_function = other.error_function
        # copy the nodes
        for node in other.nodes:
            new_node = Node(node.id, bias=node.has_bias, activation_function=node.activation_function)
            new_node.input_cons = node.input_cons.copy()
            new_node.weights = [0.5 for _ in node.input_cons]
            new_node.bias = 0.5
            new_node.activation_function = node.activation_function
            new_node.output_cons = node.output_cons.copy()
            new_node.int_val = []
            new_node.out_val = []
            this.nodes.append(new_node)
        # copy the input and output nodes
        this.input_nodes = other.input_nodes.copy()
        this.output_nodes = other.output_nodes.copy()
        this.processing_order = this.get_processing_order()
        return this
     
    def mutate(self):
        connections = self.get_connections()
        possible_connections = self.get_legal_connections()
        if (rand() < 0.5 or len(connections) == 0) and  len(possible_connections) > 0:
            # add connection
            if len(possible_connections) == 0:
                return
            
            con = choice(possible_connections) 
            self.add_connection(con[0], con[1])

        else:
            # add node
            con = choice(connections)
            self.add_node(con[0], con[1])

        self.processing_order = self.get_processing_order()

    def reset(self):
        for node in self.nodes:
            node.reset()
        self.processed_dict = {}  # Reset the processed dictionary
        self.time = 0  # Reset time if you are tracking it at the DAG level

    def __str__(self) -> str:
        return "DAG("+str([str(n) for n in self.input_nodes])+'\n\t'+'\n\t'.join(str(node) for node in self.nodes)+"\n"+str([str(n) for n in self.output_nodes])+")"

class NEAT:
    def __init__(self, nr_inputs, nr_outputs, population_size=10, error_function=MSE(), output_activation=Linear(), with_fully_connect=False):
        self.nr_inputs = nr_inputs
        self.nr_outputs = nr_outputs
        self.population_size = population_size
        self.error_function = error_function
        self.population : list[DAG] = []
        self.start_full_connect = with_fully_connect
        self.output_activation = output_activation

    def create_population(self):
        id_generator = dummy_uuid_generator()
        for _ in tqdm(range(self.population_size)):
            dag = DAG(self.nr_inputs, self.nr_outputs, id=next(id_generator), fully_connect=self.start_full_connect, error_function=self.error_function, standard_output_activation=self.output_activation)
            self.population.append(dag)

    def evaluate_population(self, inputs, targets, epochs= 100, lr=0.01, batch_size=32, verbose=False):
        #print("Evaluating population")
        population_evals = []
        for dag in tqdm(self.population, desc="evaluating DAG's"):
            losses = dag.train(inputs, targets, epochs, lr, batch_size=batch_size, verbose=verbose)
            population_evals.append([dag.id, losses[-1], losses])

        return population_evals
    
    def select_participants(self, evaluations: list, top_p=0.1):
        #print("Selecting participants")
        top_nr = max(int(len(evaluations) * top_p), 1)
        evaluations.sort(key=lambda x: x[1])
        #print("Top x", np.around([e[1] for e in evaluations[:top_nr]], 3))
        return evaluations[:top_nr], evaluations[top_nr:]
    
    def mutate(self, dags):
        for dag in dags:
            dag.mutate()

    def evolve(self, participants_id, remaining_id):
        # get the participants
        participants = [dag for dag in self.population if dag.id in [p[0] for p in participants_id]]

        # choos 50% of the remaining to be mutated
        to_mutate_ids = []
        for _ in range(len(remaining_id)//2):
            to_mutate_ids.append(remaining_id.pop(np.random.randint(len(remaining_id))))
        to_mutate = [dag for dag in self.population if dag.id in [p[0] for p in to_mutate_ids]]

        new_population = []
        # duplicate the participants
        for p in participants:
            new_population.append(DAG.from_other(p))

        # the ones to be mutated will create variations until the population size is reached
        id_generator = dummy_uuid_generator()
        for _ in range(self.population_size - len(participants)):
            parent = np.random.choice(to_mutate)
            child = DAG.from_other(parent)
            child.id = child.id + '-' + next(id_generator)
            child.mutate()
            new_population.append(child)

        return new_population

    def run(self, inputs, targets, epochs=10, lr=0.01, generations=10, batch_size=32, verbose=False): 
        self.create_population()
        self.mutate(self.population)
        best_of_generation = []
        for i in range(generations):
            print('-'*150)
            print(f"Generation {i+1}/{generations}")
            # evaluate the population
            evaluations = self.evaluate_population(inputs, targets, epochs, lr, batch_size=batch_size, verbose=verbose)
            # select the participants
            participants_id, remaining_id = self.select_participants(evaluations)
            # get the best of the generation and save it
            best = None
            best_id = participants_id[0][0]
            best_score = participants_id[0][1]
            best_losses = participants_id[0][2]

            print("Best of generation: ", best_id, "Best Score:", best_score)
            for p in self.population:
                if p.id == best_id:
                    best = [p, best_score, best_losses] 
                    break
            best_of_generation.append(best)

            if isnan(best_score):
                raise ValueError(f"Best score is NaN: {best_score}, ID: {best_id} | continue makes no sense")

            # mutate the remaining and create new population
            self.population = self.evolve(participants_id, remaining_id)

        return best_of_generation
        
def plot_dag(dag: DAG):
    dag.processing_order = dag.get_processing_order()
    # Calculate positions based on layers
    layer_positions = {}
    layer_spacing = 2  # Horizontal spacing between layers
    node_spacing = 1.5  # Vertical spacing between nodes in the same layer
    x_offset = 0

    for i, layer in enumerate(dag.processing_order):
        x_pos = x_offset * layer_spacing
        y_start = - (len(layer) - 1) * node_spacing / 2  # Center the nodes vertically

        for j, node_id in enumerate(layer):
            y_pos = y_start + j * node_spacing
            layer_positions[node_id] = (x_pos, y_pos)

        x_offset += 1  # Advance x-position for the next layer
    
    # Convert the DAG to a networkx graph
    G = nx.DiGraph()
    for node in dag.get_nodes():
        G.add_node(node.id)

    for con in dag.get_connections():
        G.add_edge(con[0], con[1])

    edge_x = []
    edge_y = []
    annotations = []
    for edge in G.edges():
        if edge[0] in layer_positions and edge[1] in layer_positions:
            x0, y0 = layer_positions[edge[0]]
            x1, y1 = layer_positions[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

            annotations.append(
                dict(
                    ax=x0, ay=y0, axref='x', ayref='y',
                    x=x1, y=y1, xref='x', yref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor='#888'
                )
            )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_color = []
    for node in G.nodes():
        if node in layer_positions:
            x, y = layer_positions[node]
            node_x.append(x)
            node_y.append(y)

            # Color the nodes based on their role
            if node in dag.input_nodes:
                node_color.append('green')
            elif node in dag.output_nodes:
                node_color.append('red')
            else:
                node_color.append('orange')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        textposition='top center',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=node_color,
            size=20,
            colorbar=dict(
                thickness=15,
                title='Node Type',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_text = []
    for node in dag.get_nodes():
        node_text.append(f'Node {node.id}<br>Out Val: {node.spike}')  # Include the out_val in the text

    node_trace.text = node_text

    # Draw circles around nodes in the same M-Layer
    layer_shapes = []
    for layer in dag.processing_order:
        if isinstance(layer, MLayer):
            m_layer_positions = np.array([layer_positions[node_id] for node_id in layer if node_id in layer_positions])
            if len(m_layer_positions) > 0:
                x_center = np.mean(m_layer_positions[:, 0])
                y_center = np.mean(m_layer_positions[:, 1])
                x_range = np.ptp(m_layer_positions[:, 0])  # Peak-to-peak (max-min) range in x
                y_range = np.ptp(m_layer_positions[:, 1])  # Peak-to-peak (max-min) range in y

                # Add some padding to the circle
                padding = 0.1
                width = x_range + padding
                height = y_range + padding

                # Append the shape for the M-Layer
                layer_shapes.append(
                    dict(
                        type="circle",
                        xref="x", yref="y",
                        x0=x_center - width / 2,
                        y0=y_center - height / 2,
                        x1=x_center + width / 2,
                        y1=y_center + height / 2,
                        line=dict(color="rgba(255, 0, 0, 0.5)", width=2, dash="solid")
                    )
                )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network graph of a DAG',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=annotations,
                        #shapes=layer_shapes,  # Add the layer shapes to the layout
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )

    fig.show()
 