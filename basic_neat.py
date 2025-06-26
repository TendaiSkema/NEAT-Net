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

def dummy_uuid_generator():
    i = 0
    while True:
        yield str(i)
        i += 1

dummy_gen = dummy_uuid_generator()

def uuid1():
    return next(dummy_gen)

class Sigmoid:
    def __call__(self, x):
        return 1/(1+np.exp(-x))
    
    def derivative(self, x):
        return self(x)*(1-self(x))

class ReLU:
    def __call__(self, x):
        return np.where(x > 0, x, 0)
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)
    
class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def __call__(self, x):
        return np.where(x > 0, x, self.alpha*x)
    
    def derivative(self, x):
        return np.where(x > 0, 1, self.alpha)
    
class Elu:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, x):
        return np.where(x > 0, x, self.alpha*(np.exp(x)-1))
    
    def derivative(self, x):
        return np.where(x > 0, 1, self(x) + self.alpha)

class Tanh:
    def __call__(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - np.tanh(x)**2
    
class Linear:
    def __call__(self, x):
        return x
    
    def derivative(self, x):
        return np.ones_like(x)

class MSE:
    def __call__(self, y, t):
        return np.sum((y - t)**2)/len(y)
    
    def derivative(self, y, t):
        return - (t - y)
    
class CrossEntropy:
    def __call__(self, y, t):
        return -np.sum(t*np.log(y) + (1-t)*np.log(1-y))
    
    def derivative(self, y, t):
        return (y-t)/(y*(1-y))
    
class Softmax:
    def __call__(self, x):
        return np.exp(x)/np.sum(np.exp(x))
    
    def derivative(self, x):
        return self(x)*(1-self(x))

class Node:
    def __init__(self, id, bias=True, activation_function=LeakyReLU()):
        self.id = id
        self.input_cons = []
        self.weights = np.array([])
        self.has_bias = bias
        self.bias = 0 #np.random.random() if bias else 0  # Small random value if bias
        self.activation_function = activation_function
        self.output_cons = []
        self.int_val = []
        self.out_val = []

    def add_input_con(self, id):
        self.input_cons.append(id)
        wights = list(self.weights)
        wights.append(np.random.random() * 2 - 1)  # Small random values
        self.weights = np.array(wights)

    def add_output_con(self, id):
        self.output_cons.append(id)

    def remove_input_con(self, id):
        idx = self.input_cons.index(id)
        w = list(self.weights)
        w.pop(idx)
        self.weights = np.array(w)
        self.input_cons.remove(id)

    def remove_output_con(self, id):
        self.output_cons.remove(id)

    def set_internal_value(self, val):
        self.int_val = val
        self.out_val = val

    def process(self, inputs=[]):
        if len(inputs):
            self.int_val = np.dot(inputs, self.weights) + self.bias
        if self.activation_function:
            self.out_val = self.activation_function(self.int_val)
        else:
            self.out_val = self.int_val

    def backprop(self, errors, input_values, lr) -> list:
        # internal error
        e_int = errors * self.activation_function.derivative(self.int_val)

        # update bias
        if self.has_bias:
            self.bias += lr * np.sum(e_int)
        # update weights and return errors for input nodes
        prop_e = []
        #print(f"\tUpdating weights: {self.weights}")
        for i, n_id in enumerate(self.input_cons):
            prop_e.append((n_id, np.dot(e_int, self.weights[i])))
            self.weights[i] += lr * np.sum(np.dot(input_values[i], e_int))

        return prop_e

    def reset(self):
        self.int_val = []
        self.out_val = []

    def __str__(self) -> str:
        return f"Node({self.id}: in: {[str(n) for n in self.input_cons]} out: {[str(n) for n in self.output_cons]})"

class DAG: 
    def __init__(self, nr_inputs, nr_outputs, id='0', do_setup=True, fully_connect=False, 
                error_function=MSE(), 
                standard_input_activation=None, standard_output_activation=Linear(), standard_hidden_activation=LeakyReLU()
                ):
        self.id = id
        self.nodes = []
        self.input_nodes = []
        self.output_nodes = []
        self.processing_order = []
        self.nr_inputs = nr_inputs
        self.nr_outputs = nr_outputs
        self.error_function = error_function
        self.standard_hidden_activation = standard_hidden_activation

        if do_setup:
            for i in range(nr_inputs):
                self.nodes.append(Node(uuid1(), activation_function=standard_input_activation))
                self.input_nodes.append(self.nodes[-1].id)
            
            for i in range(nr_outputs):
                self.nodes.append(Node(uuid1(), activation_function=standard_output_activation))
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
        '''
        check if node2 is reachable from node1 using a depth first search
        '''
        connections = self.get_connections()
        new_edge = (nod1_id, nod2_id)
    
        # Convert connections to adjacency list
        graph = defaultdict(list)
        for start, end in connections:
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
                if self.is_connected(n1.id, n2.id) or self.is_connected(n2.id, n1.id):
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
        
        activ_func = activation_function if activation_function else self.standard_hidden_activation
        new_node = Node(uuid1(), activation_function=activ_func)
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

    def get_processing_order(self) -> list:
        processing_order = []
        nodes = self.get_nodes_as_dict()
        # while there is still nodes to process
        while len(nodes) > 0:
            # go over all nodes
            new_layer = []
            for node_id, node in nodes.items():
                # if node has no input connections
                if len(node["input_cons"]) == 0:
                    # add to layer
                    new_layer.append(node_id)

            processing_order.append(new_layer)
            
            # remove node from other nodes input connections
            for node_id in new_layer:
                for out in nodes[node_id]["output_cons"]:
                    nodes[out]["input_cons"].remove(node_id)
                del nodes[node_id]
            

        return processing_order
    
    def process(self, inputs, verbose=False):
        # get the processing order
        processed_dict = {}
        # set all inputs to the input nodes
        inputs = np.array(inputs)
        for i, node_id in enumerate(self.input_nodes):
            node = self.get_node(node_id)
            node_inputs = inputs[:,i]

            node.set_internal_value(node_inputs)
            processed_dict[node_id] = node.out_val

        for layer in self.processing_order:
            for node_id in layer:
                if node_id in self.input_nodes:
                    continue
                
                node = self.get_node(node_id)

                # get all the values that the node has as input that are already processed
                node_inputs = []
                for in_node_id in node.input_cons:
                    new_input = processed_dict[in_node_id]
                    node_inputs.append(new_input)

                node_inputs = np.array(node_inputs)
                
                # process the node
                node.process(node_inputs.T)
                # save the output value
                processed_dict[node_id] = node.out_val
                
        # return the output of the output nodes
        return np.array([self.get_node(node).out_val for node in self.output_nodes]).T

    def cost(self, outputs, targets):
        return self.error_function(np.array(outputs), np.array(targets))

    def backprop(self, outputs, targets, lr): 
        error_dict = {}
        cost = self.cost(outputs, targets)
        # create error for output nodes
        for node_id in self.output_nodes:
            node = self.get_node(node_id)
            error = self.error_function.derivative(node.out_val, np.array(targets)[:, self.output_nodes.index(node.id)])
            error_dict[node_id] = error

        for layer in reversed(self.processing_order):
            for node_id in layer:
                # skip input nodes
                if node_id in self.input_nodes:
                    continue
                
                node = self.get_node(node_id)
                
                # get the error for the node
                error = error_dict[node_id]

                # get the input values of the node
                input_vals = np.array([self.get_node(n_id).out_val for n_id in node.input_cons])

                # backpropagate the error
                prop_err = node.backprop(error, input_vals, lr)
                # add the propagated error to the error dict
                for n_id, e in prop_err:
                    if n_id in error_dict:
                        error_dict[n_id] += e
                    else:
                        error_dict[n_id] = e

        return cost
 

    def train(self, inputs, targets, epochs=100, lr=0.01, batch_size=None, verbose=False):
        if verbose:
            print(f"Start Training DAG {self.id} with", epochs, "epochs")
        if batch_size is None:
            batch_size = len(inputs)
        losses = []
        self.processing_order = self.get_processing_order()
        iterator = tqdm(range(epochs)) if verbose else range(epochs)
        for _ in iterator:
            nr_batches = len(inputs) // batch_size
            if len(inputs) % batch_size != 0:
                nr_batches += 1
            for i in range(nr_batches):
                if i == nr_batches - 1:
                    in_batch = inputs[i * batch_size:]
                    tar_batch = targets[i * batch_size:]
                else:
                    in_batch = inputs[i * batch_size:(i + 1) * batch_size]
                    tar_batch = targets[i * batch_size:(i + 1) * batch_size]

                outputs = self.process(in_batch)
                loss = self.backprop(outputs, tar_batch, lr)
                losses.append(loss)
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

    def __str__(self) -> str:
        return "DAG("+str([str(n) for n in self.input_nodes])+'\n\t'+'\n\t'.join(str(node) for node in self.nodes)+"\n"+str([str(n) for n in self.output_nodes])+")"

class NEAT:
    def __init__(self, nr_inputs, nr_outputs, population_size=10, error_function=MSE(), output_activation=Linear(), with_fully_connect=False):
        self.nr_inputs = nr_inputs
        self.nr_outputs = nr_outputs
        self.population_size = population_size
        self.error_function = error_function
        self.population = []
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

            # mutate the remaining and create new population
            self.population = self.evolve(participants_id, remaining_id)

        return best_of_generation
        
def plot_dag(dag: DAG):
    # Convert the DAG to a networkx graph
    G = nx.DiGraph()
    for node in dag.get_nodes():
        G.add_node(node.id)

    for con in dag.get_connections():
        G.add_edge(con[0], con[1])

    # Generate plotly figure with spring layout
    pos = nx.spring_layout(G)  # spring layout for our nodes

    edge_x = []
    edge_y = []
    #edge_weights = []
    annotations = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

        # Get the weight of the edge
        source_node = dag.get_node(edge[0])
        target_node = dag.get_node(edge[1])
        """ weight_index = source_node.output_cons.index(target_node.id)
        weight = source_node.weights[weight_index] """

        #edge_weights.append(weight)
        
        # Add an annotation for the arrow
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
        
        # Add annotation for the weight
        """ annotations.append(
            dict(
                x=(x0 + x1) / 2, y=(y0 + y1) / 2, xref='x', yref='y',
                text=f'{weight:.2f}', showarrow=False,
                font=dict(color='red')
            )
        )"""

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
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
        node_text.append(f'Node {node.id}<br>Out Val: {node.out_val}')  # Include the out_val in the text

    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Network graph of a DAG',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=annotations,
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False))
                    )

    fig.show()

