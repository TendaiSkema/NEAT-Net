package main

import (
	"fmt"
	"math"
	"math/rand"
)

type ActivationFunc interface {
	Activate(x float64) float64
	Derivative(x float64) float64
	Name() string
}

// / Sigmoid Activation Function
type sigmoidActivation struct{}

func (sigmoidActivation) Activate(x float64) float64 {
	return 1 / (1 + float64(math.Exp(-x)))
}

func (sigmoidActivation) Derivative(x float64) float64 {
	sig := 1 / (1 + float64(math.Exp(-x)))
	return sig * (1 - sig)
}
func (sigmoidActivation) Name() string {
	return "sigmoid"
}

// / Relu activation Function
type reluActivation struct{}

func (reluActivation) Activate(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}
func (reluActivation) Derivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}
func (reluActivation) Name() string {
	return "relu"
}

// / Linear activation Function
type nullActivation struct{}

func (nullActivation) Activate(x float64) float64 {
	panic("null activation should never be triggered")
}
func (nullActivation) Derivative(x float64) float64 {
	panic("null activation should never be triggered")
}
func (nullActivation) Name() string {
	return "null"
}

// / Node in the DAG
type Node struct {
	Id             string     `json:"id"`
	InputNodeIds   []string   `json:"inputNodeIds"`
	Inputs         []*float64 `json:"inputs"`
	Weights        []float64  `json:"weights"`
	Output         float64    `json:"output"`
	activationFunc ActivationFunc
}

func NewNode(id string, activation ActivationFunc) Node {
	node := Node{
		Id:             id,
		InputNodeIds:   make([]string, 0),
		Inputs:         make([]*float64, 0),
		Weights:        make([]float64, 0),
		activationFunc: activation,
	}
	return node
}

func (n *Node) AddInput(inputNode *Node) {
	n.Inputs = append(n.Inputs, &inputNode.Output)
	weight := rand.Float64()*2 - 1 // Random weight between -1 and 1
	n.Weights = append(n.Weights, weight)
	n.InputNodeIds = append(n.InputNodeIds, inputNode.Id)
	fmt.Printf("Connected %p to %p with weight %f Inputs now: %v\n", inputNode, n, weight, n.InputNodeIds)
}

func (n *Node) Forward() {
	if len(n.Inputs) == 0 {
		// Input node, output is set externally
		return
	}
	sum := 0.0
	for i, inputPtr := range n.Inputs {
		sum += (*inputPtr) * n.Weights[i]
	}
	n.Output = n.activationFunc.Activate(sum)
}

func (n Node) String() string {
	return fmt.Sprintf("Node(ID: %s, Inputs: %v, Activation: %s)", n.Id, n.InputNodeIds, n.activationFunc.Name())
}

func (n Node) DetailedString() string {
	return fmt.Sprintf("Node(ID: %s, Inputs: %v, Weights: %v, Output: %.4f, Activation: %s)", n.Id, n.InputNodeIds, n.Weights, n.Output, n.activationFunc.Name())
}

// / Directed Acyclic Graph (DAG)
type DAG struct {
	Id          int              `json:"id"`
	Nodes       map[string]*Node `json:"nodes"`
	InputNodes  []*Node          `json:"inputNodes"`
	OutputNodes []*Node          `json:"outputNodes"`
	Inputs      []*float64       // Pointers to input values to set directly
	Outputs     []*float64       // Pointers to output values to read directly
}

func NewDAG(inputs int, outputs int, outActivation ActivationFunc, fullyConnect bool) DAG {
	DAG := DAG{
		Id:          rand.Int(),
		Nodes:       make(map[string]*Node),
		InputNodes:  make([]*Node, inputs),
		OutputNodes: make([]*Node, outputs),
		Inputs:      make([]*float64, inputs),
		Outputs:     make([]*float64, outputs),
	}
	for i := 0; i < inputs; i++ {
		inputNode := NewNode(fmt.Sprintf("input_%d", i), nullActivation{})
		DAG.InputNodes[i] = &inputNode
		DAG.Nodes[inputNode.Id] = &inputNode
		DAG.Inputs[i] = &inputNode.Output
		fmt.Printf("Created input node: %p\n", &inputNode)
	}
	for i := 0; i < outputs; i++ {
		outputNode := NewNode(fmt.Sprintf("output_%d", i), outActivation)
		DAG.OutputNodes[i] = &outputNode
		DAG.Nodes[outputNode.Id] = &outputNode
		DAG.Outputs[i] = &outputNode.Output
		fmt.Printf("Created output node: %p\n", &outputNode)
	}

	if fullyConnect {
		for _, inNode := range DAG.InputNodes {
			for _, outNode := range DAG.OutputNodes {
				fmt.Printf("Connecting %p to %p\n", inNode, outNode)
				outNode.AddInput(inNode)
			}
		}
	}

	return DAG
}

func (dag *DAG) AddNode(node Node) {
	dag.Nodes[node.Id] = &node
}

func (dag *DAG) ConnectNodes(fromNodeId, toNodeId string) error {
	fromNode, fromExists := dag.Nodes[fromNodeId]
	toNode, toExists := dag.Nodes[toNodeId]
	if !fromExists || !toExists {
		return fmt.Errorf("one or both nodes do not exist")
	}
	toNode.AddInput(fromNode)
	return nil
}

func (dag *DAG) SetInputs(inputValues []float64) error {
	if len(inputValues) != len(dag.Inputs) {
		return fmt.Errorf("input values length mismatch")
	}
	for i, val := range inputValues {
		*dag.Inputs[i] = val
	}
	return nil
}

func (dag *DAG) GetOutputs() []float64 {
	output := make([]float64, len(dag.Outputs))
	for i, outPtr := range dag.Outputs {
		output[i] = *outPtr
	}
	return output
}

func (dag *DAG) Forward() {
	// Forward pass through all nodes
	for _, node := range dag.Nodes {
		node.Forward()
	}
}

func (dag DAG) String() string {
	result := "DAG:\n"
	for _, node := range dag.Nodes {
		result += "\t" + fmt.Sprintf("%p", node) + " " + node.String() + "\n"
	}
	return result
}

func (dag DAG) DetailedString() string {
	result := "DAG Detailed:\n"
	for _, node := range dag.Nodes {
		result += node.DetailedString() + "\n"
	}
	return result
}

func main() {
	dag := NewDAG(2, 1, sigmoidActivation{}, true)
	fmt.Println(dag.String())
}
