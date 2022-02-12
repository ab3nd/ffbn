#!/usr/bin/env python3

# First cut at implementing a version of NEAT (https://www.cs.ucf.edu/~kstanley/neat.html)
# but with a few differences to make it work for creating a feed-forward boolean network. 
# The differences between a neural network as constructed by NEAT and a FFBN are:
# 1. Weights are effectively boolean. There's either a wire there (1) or there's not (0).
#    This probably means that mutating "weights" has huge impacts, effectively cutting out or 
#    adding new sections to the network instantly. Initially, weight mutation won't be a thing. 
#
# 2. Nodes are boolean operations (and, or, not). Could also explictly add secondary operations 
#    (nand, nor, xor, not), but I'm not certain it would add anything. This brings in the complication
#    that while a typical ANN node can have any number of inputs and handle them with summation, 
#    boolean AND and OR have 2 or more inputs, while NOT has exactly one input. 
#
#    To handle NOT, I'm going to not handle it. The set of available nodes can be AND, OR, NAND, NOR, 
#    or we can get silly about this and use _only_ NAND, since it is a universal gate. 
#
#    Handling all nodes having at least two inputs while preserving feed-forwardness can be done by 
#    implementing "add node" just like NEAT (pick a connection, add the node in the middle of it) and 
#    wiring the other input of the gate/node to a bias (or saying it floats to a particular value). That 
#    will render the gate either transparent (AND with 1), inverting (NAND with 1), or blocking (AND with 0), 
#    which is probably going to have a major downstream impact (or none, if transparent). 
#
#    It could also be handled by following every "add node" with an "add edge" that has to target the 
#    disconnected input of the gate. This is a better solution, in that changes to the genome hav an
#    immediate effect on the phenotype instead of potentially doing nothing. 

from enum import Enum
import random

class NodeType(Enum):
    AND = 1
    OR = 2
    NAND = 3
    NOR = 4 
    INPUT = 5 # Special class for representing inputs, can't have connections going to it
    OUTPUT = 6 # Special class for representing the output, can't have connections coming from it

class Node(object):
    def __init__(self, innovation, node_type = None):
        self.innovation = innovation
        self.node_type = node_type
        if self.node_type is None:
            # Pick a random type that's not input or output
            self.node_type = random.choice([NodeType.AND, NodeType.OR, NodeType.NAND, NodeType.NOR])
        
        def get_type():
            return self.node_type
        
        def get_innovation():
            return self.innovation

class Connection(object):
    def __init__(self, input, output, enabled = True, innovation):
        self.input = input
        self.output = output
        self.enabled = enabled
        self.innovation = innovation

class Genome(object):
    def __init__(self, input_count, output_count):
        self.connections = []
        self.nodes = []
        self.innovation_counter = 0

        # Generate input and output nodes
        for ii in range(input_count):
            self.nodes.append(Node(self.innovation_counter, NodeType.INPUT))
            self.innovation_counter += 1

        for ii in range(output_count):
            self.nodes.append(Node(self.innovation_counter, NodeType.OUTPUT))
            self.innovation_counter += 1

        # Connect some of the inputs directly to outputs
        # NOTE outputs are effectively wires, and so having two connections to an output doesn't
        # make a lot of sense (for the same reasons connecting two gate outputs together is a bad move)
        # NOTE Since innovations are unique within a genome, they can be used as node IDs
        inputs = list(filter(lambda node: node.get_type() is NodeType.INPUT, self.nodes))
        random.shuffle(inputs)
        outputs = list(filter(lambda node: node.get_type() is NodeType.OUTPUT, self.nodes))
        random.shuffle(outputs)
        while True:
            try: 
                from_node = inputs.pop()
                to_node = outputs.pop()
                self.connections.append(Connection(from_node.get_innovation(), to_node.get_innovation(), self.innovation_counter))
                self.innovation_counter += 1
            except IndexError:
                # One of the lists ran out
                break



    # Implements the "add connection" mutation. Connect two nodes that 
    # are not currently connected, preserving feed-forwardness
    def add_connection():
        pass

    # Implements the "add node" mutation. Select a connection, deactivate it, replace it with a node
    # that has the connection's inputs and outputs. Since this is for 2+ input gates, also connect
    # the other input of the gate to some node's output, preserving feed-forwardness. 
    def add_node():
        pass


# Let's try implementing XOR. 
#
# XOR truth table   
# in  | out            
# ---------
# 0 0 | 0
# 0 1 | 1
# 1 0 | 1
# 1 1 | 0
# For the first generation, it's impossible that any of the nodes will be XOR. 
# In fact, they will all be some version of a straight wire between one input and the output, 
# and so will all be right for 0.25 of the truth table, or show 0.75 error. The first wave of mutation 
# and crossover will be the one that starts actually doing anything. 

# TODO think some about what a proper representation is. It's possible to build a whole representation of 
# e.g. MNIST in this format
xor = {[0,0]:[0],
       [0,1]:[1],
       [1,0]:[1],
       [1,1]:[1]}

def evaluate(genome, truth):
    # TODO convert genome into a boolean expression
    # For each output, do a depth first traversal of the tree rooted at the output. The inputs are the 
    # leaves of the tree. Return an expression that can be evaluated with eval or whatever. 
    # Evaluate the expressions in order, save the values to a list and return that. 

    score = 0
    for key in truth.keys():
        # If applying the key to the boolean expression matches the result, 
        # add one to the score, otherwise don't add to the score
        pass
    
    #Score is how many rows this expression got right out of a truth table
    return score/len(truth)
if __name__ == "__main__":
    input_count = 2
    output_count = 1
    total_population = 100
    population = []
    # Generate a lot of genomes with two inputs and one output
    for ii in range(total_population):
        population.append(Genome(input_count, output_count))

    # For each genome, evaluate it
    scores = []
    for g in population:
        scores.append(evaluate(g, xor))
