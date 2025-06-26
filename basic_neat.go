package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

func dummyUUIDGenerator() func() string {
	i := 0
	return func() string {
		i++
		return fmt.Sprintf("%d", i)
	}
}

var uuid1 = dummyUUIDGenerator()

type ActivationFunction interface {
	Call(x float64) float64
	Derivative(x float64) float64
}

type Sigmoid struct{}

func (s Sigmoid) Call(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (s Sigmoid) Derivative(x float64) float64 {
	return s.Call(x) * (1 - s.Call(x))
}

type ReLU struct{}

func (r ReLU) Call(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func (r ReLU) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

type Elu struct {
	Alpha float64
}

func (e Elu) Call(x float64) float64 {
	if x > 0 {
		return x
	}
	return e.Alpha * (math.Exp(x) - 1)
}

func (e Elu) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return e.Call(x) + e.Alpha
}

type Tanh struct{}

func (t Tanh) Call(x float64) float64 {
	return math.Tanh(x)
}

func (t Tanh) Derivative(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
}

type Linear struct{}

func (l Linear) Call(x float64) float64 {
	return x
}

func (l Linear) Derivative(x float64) float64 {
	return 1
}

type MSE struct{}

func (m MSE) Call(y, t float64) float64 {
	return math.Pow(y-t, 2) / 2
}

func (m MSE) Derivative(y, t float64) float64 {
	return y - t
}

type CrossEntropy struct{}

func (c CrossEntropy) Call(y, t float64) float64 {
	return -(t*math.Log(y) + (1-t)*math.Log(1-y))
}

func (c CrossEntropy) Derivative(y, t float64) float64 {
	return (y - t) / (y * (1 - y))
}

type Softmax struct{}

func (s Softmax) Call(x float64) float64 {
	return math.Exp(x) / math.Exp(x) // This needs to be implemented correctly
}

func (s Softmax) Derivative(x float64) float64 {
	return s.Call(x) * (1 - s.Call(x))
}

type Node struct {
	ID                 string
	InputConnections   []string
	Weights            []float64
	HasBias            bool
	Bias               float64
	ActivationFunction ActivationFunction
	OutputConnections  []string
	InternalValue      float64
	OutputValue        float64
}

func NewNode(id string, bias bool, activationFunction ActivationFunction) *Node {
	return &Node{
		ID:                 id,
		HasBias:            bias,
		Bias:               rand.Float64()*2 - 1,
		ActivationFunction: activationFunction,
	}
}

func (n *Node) AddInputConnection(id string) {
	n.InputConnections = append(n.InputConnections, id)
	n.Weights = append(n.Weights, rand.Float64()*2-1)
}

func (n *Node) AddOutputConnection(id string) {
	n.OutputConnections = append(n.OutputConnections, id)
}

func (n *Node) RemoveInputConnection(id string) {
	idx := -1
	for i, v := range n.InputConnections {
		if v == id {
			idx = i
			break
		}
	}
	if idx != -1 {
		n.InputConnections = append(n.InputConnections[:idx], n.InputConnections[idx+1:]...)
		n.Weights = append(n.Weights[:idx], n.Weights[idx+1:]...)
	}
}

func (n *Node) RemoveOutputConnection(id string) {
	idx := -1
	for i, v := range n.OutputConnections {
		if v == id {
			idx = i
			break
		}
	}
	if idx != -1 {
		n.OutputConnections = append(n.OutputConnections[:idx], n.OutputConnections[idx+1:]...)
	}
}

func (n *Node) SetInternalValue(val float64) {
	n.InternalValue = val
	n.OutputValue = val
}

func (n *Node) Process(inputs []float64) {
	sum := 0.0
	for i, input := range inputs {
		sum += input * n.Weights[i]
	}
	if n.HasBias {
		sum += n.Bias
	}
	n.InternalValue = sum
	n.OutputValue = n.ActivationFunction.Call(sum)
}

func (n *Node) Backprop(errors, inputValues []float64, lr float64) []float64 {
	eInt := errors[0] * n.ActivationFunction.Derivative(n.InternalValue)
	if n.HasBias {
		n.Bias -= lr * eInt
	}
	propE := make([]float64, len(n.InputConnections))
	for i, _ := range n.InputConnections {
		propE[i] = n.Weights[i] * eInt
		n.Weights[i] -= lr * eInt * inputValues[i]
	}
	return propE
}

type DAG struct {
	ID                       string
	Nodes                    []*Node
	InputNodes               []string
	OutputNodes              []string
	ProcessingOrder          [][]string
	NrInputs                 int
	NrOutputs                int
	ErrorFunction            MSE
	StandardHiddenActivation ActivationFunction
}

func NewDAG(nrInputs, nrOutputs int, id string, fullyConnect bool) *DAG {
	dag := &DAG{
		ID:                       id,
		NrInputs:                 nrInputs,
		NrOutputs:                nrOutputs,
		ErrorFunction:            MSE{},
		StandardHiddenActivation: ReLU{},
	}

	for i := 0; i < nrInputs; i++ {
		node := NewNode(uuid1(), false, Linear{})
		dag.Nodes = append(dag.Nodes, node)
		dag.InputNodes = append(dag.InputNodes, node.ID)
	}

	for i := 0; i < nrOutputs; i++ {
		node := NewNode(uuid1(), false, Linear{})
		dag.Nodes = append(dag.Nodes, node)
		dag.OutputNodes = append(dag.OutputNodes, node.ID)
	}

	if fullyConnect {
		for _, inputNode := range dag.InputNodes {
			for _, outputNode := range dag.OutputNodes {
				if inputNode != outputNode {
					dag.AddConnection(inputNode, outputNode)
				}
			}
		}
	}

	dag.ProcessingOrder = dag.GetProcessingOrder()

	return dag
}

func (dag *DAG) GetNode(id string) *Node {
	for _, node := range dag.Nodes {
		if node.ID == id {
			return node
		}
	}
	return nil
}

func (dag *DAG) GetConnections() [][2]string {
	var connections [][2]string
	for _, node := range dag.Nodes {
		for _, outputCon := range node.OutputConnections {
			connections = append(connections, [2]string{node.ID, outputCon})
		}
	}
	return connections
}

func (dag *DAG) IsConnected(nod1ID, nod2ID string) bool {
	node1 := dag.GetNode(nod1ID)
	for _, outputCon := range node1.OutputConnections {
		if outputCon == nod2ID {
			return true
		}
	}
	return false
}

func (dag *DAG) AddNode(node1ID, node2ID string, activationFunction ActivationFunction) bool {
	if !dag.IsConnected(node1ID, node2ID) {
		return false
	}
	newNode := NewNode(uuid1(), false, activationFunction)
	dag.Nodes = append(dag.Nodes, newNode)
	dag.RemoveConnection(node1ID, node2ID)
	dag.AddConnection(node1ID, newNode.ID)
	dag.AddConnection(newNode.ID, node2ID)
	return true
}

func (dag *DAG) AddConnection(nod1ID, nod2ID string) bool {
	if dag.IsConnected(nod1ID, nod2ID) {
		return false
	}
	node1 := dag.GetNode(nod1ID)
	node2 := dag.GetNode(nod2ID)
	node1.AddOutputConnection(node2.ID)
	node2.AddInputConnection(node1.ID)
	return true
}

func (dag *DAG) RemoveConnection(nod1ID, nod2ID string) {
	node1 := dag.GetNode(nod1ID)
	node2 := dag.GetNode(nod2ID)
	node1.RemoveOutputConnection(nod2ID)
	node2.RemoveInputConnection(nod1ID)
}

func (dag *DAG) Process(inputs []float64) []float64 {
	processedDict := make(map[string]float64)
	for i, nodeID := range dag.InputNodes {
		node := dag.GetNode(nodeID)
		node.SetInternalValue(inputs[i])
		processedDict[nodeID] = node.OutputValue
	}

	for _, layer := range dag.ProcessingOrder {
		for _, nodeID := range layer {
			if contains(dag.InputNodes, nodeID) {
				continue
			}
			node := dag.GetNode(nodeID)
			var inputVals []float64
			for _, inputCon := range node.InputConnections {
				inputVals = append(inputVals, processedDict[inputCon])
			}
			node.Process(inputVals)
			processedDict[nodeID] = node.OutputValue
		}
	}

	var outputs []float64
	for _, nodeID := range dag.OutputNodes {
		outputs = append(outputs, dag.GetNode(nodeID).OutputValue)
	}
	return outputs
}

func (dag *DAG) GetProcessingOrder() [][]string {
	var processingOrder [][]string
	nodes := make(map[string]Node)
	for _, node := range dag.Nodes {
		nodes[node.ID] = *node
	}
	for len(nodes) > 0 {
		var newLayer []string
		for nodeID, node := range nodes {
			if len(node.InputConnections) == 0 {
				newLayer = append(newLayer, nodeID)
			}
		}
		processingOrder = append(processingOrder, newLayer)
		for _, nodeID := range newLayer {
			for _, outputCon := range nodes[nodeID].OutputConnections {
				node := nodes[outputCon]
				node.InputConnections = remove(node.InputConnections, nodeID)
				nodes[outputCon] = node
			}
			delete(nodes, nodeID)
		}
	}
	return processingOrder
}

func (dag *DAG) Backprop(errors, inputValues []float64, lr float64) {
	for i := len(dag.ProcessingOrder) - 1; i >= 0; i-- {
		for _, nodeID := range dag.ProcessingOrder[i] {
			if contains(dag.InputNodes, nodeID) {
				continue
			}
			node := dag.GetNode(nodeID)
			propE := node.Backprop(errors, inputValues, lr)
			for _, inputCon := range node.InputConnections {
				inputNode := dag.GetNode(inputCon)
				inputNode.Backprop(propE, inputValues, lr)
			}
		}
	}
}

func remove(slice []string, s string) []string {
	for i, v := range slice {
		if v == s {
			return append(slice[:i], slice[i+1:]...)
		}
	}
	return slice
}

func contains(slice []string, s string) bool {
	for _, v := range slice {
		if v == s {
			return true
		}
	}
	return false
}

type NEAT struct {
	NrInputs         int
	NrOutputs        int
	PopulationSize   int
	Population       []*DAG
	StartFullConnect bool
}

func NewNEAT(nrInputs, nrOutputs, populationSize int, startFullConnect bool) *NEAT {
	return &NEAT{
		NrInputs:         nrInputs,
		NrOutputs:        nrOutputs,
		PopulationSize:   populationSize,
		StartFullConnect: startFullConnect,
	}
}

func (n *NEAT) CreatePopulation() {
	for i := 0; i < n.PopulationSize; i++ {
		dag := NewDAG(n.NrInputs, n.NrOutputs, uuid1(), n.StartFullConnect)
		n.Population = append(n.Population, dag)
	}
}

func (n *NEAT) EvaluatePopulation(inputs, targets []float64, epochs int, lr float64) []float64 {
	var wg sync.WaitGroup
	evaluations := make([]float64, n.PopulationSize)
	for i, dag := range n.Population {
		wg.Add(1)
		go func(i int, dag *DAG) {
			defer wg.Done()
			for epoch := 0; epoch < epochs; epoch++ {
				outputs := dag.Process(inputs)
				errors := make([]float64, len(outputs))
				for j, output := range outputs {
					errors[j] = dag.ErrorFunction.Derivative(output, targets[j])
				}
				dag.Backprop(errors, inputs, lr)
				evaluations[i] = dag.ErrorFunction.Call(outputs[0], targets[0])
			}
		}(i, dag)
	}
	wg.Wait()
	return evaluations
}

func main() {
	rand.Seed(time.Now().UnixNano())

	inputs := []float64{0.5, 0.3}
	targets := []float64{0.7}
	neat := NewNEAT(2, 1, 10, false)
	neat.CreatePopulation()
	evaluations := neat.EvaluatePopulation(inputs, targets, 100, 0.01)

	for _, eval := range evaluations {
		fmt.Println(eval)
	}
}
