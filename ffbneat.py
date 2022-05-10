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
    def __init__(self, innovation, layer = 0.0, node_type = None):
        self.innovation = innovation
        self.node_type = node_type
        self.layer = layer
        if self.node_type is None:
            # Pick a random type that's not input or output
            self.node_type = random.choice([NodeType.AND, NodeType.OR, NodeType.NAND, NodeType.NOR])
        elif self.node_type is NodeType.OUTPUT:
            self.layer = 1.0
        elif self.node_type is NodeType.INPUT:
            self.layer = 0.0

    def get_type(self):
        return self.node_type
    
    def get_innovation(self):
        return self.innovation

class Connection(object):
    def __init__(self, input, output, innovation, enabled = True):
        self.input = input
        self.output = output
        self.enabled = enabled
        self.innovation = innovation

class Genome(object):
    # TODO in order to maintain feed-forward condition, the genome is going to have to organize 
    # the nodes into layers, and only connect from nodes of lower layers to nodes of higher 
    # layers. The actual layers themselves are pretty arbitrary, as long as the ordering is enforced. 
    # Since I kind of want arbitrary layers, the layer is floating point, and a new node has a layer
    # value of the average of its input and output's layers. 
    def __init__(self, input_count, output_count):
        self.connections = []
        self.nodes = []
        self.innovation_counter = 0

        # Generate input and output nodes
        for ii in range(input_count):
            self.nodes.append(Node(self.innovation_counter, node_type=NodeType.INPUT))
            self.innovation_counter += 1

        for ii in range(output_count):
            self.nodes.append(Node(self.innovation_counter, node_type=NodeType.OUTPUT))
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
    def add_connection(self):
        # Cannot connect multiple outputs to one output node, that's basically wiring digital outputs
        # to each other, which can lead to short circuits if they're driven to different levels. 
        valid_nodes = list(filter(lambda node: node.get_type() is not NodeType.OUTPUT, self.nodes))
        
        # Check that they're not the same layer
        got_connection = False
        while not got_connection:
            node_a, node_b = random.sample(valid_nodes, 2)
            # Ensure they're not in the same layer
            if node_a.layer > node_b.layer:
                new_conn = Connection(node_b.get_innovation(), node_a.get_innovation(), self.innovation_counter)
            elif node_a.layer < node_b.layer:
                new_conn = Connection(node_a.get_innovation(), node_b.get_innovation(), self.innovation_counter)
            else:
                continue

            # Check if the connection already exists
            got_connection = True
            for conn in self.connections:
                if conn.input == new_conn.input and conn.output == new_conn.output:
                    # This is effectively the same connection
                    got_connection = False
                    break
            
            # Ok, passed both checks, so keep the connection and increment innovation
            if got_connection:
                self.connections.append(new_conn)
                self.innovation_counter += 1
        

    # Implements the "add node" mutation. Select a connection, deactivate it, replace it with a node
    # that has the connection's inputs and outputs. Since this is for 2+ input gates, also connect
    # the other input of the gate to some node's output, preserving feed-forwardness. 
    def add_node(self):
        # Pick a random connection and deactivate it
        conn = random.choice([c for c in self.connections if c.enabled])
        conn.enabled = False
        
        # Get the layer between the two things the connection used to connect
        in_layer = [n for n in self.nodes if n.get_innovation() == conn.input][0].layer
        out_layer = [n for n in self.nodes if n.get_innovation() == conn.output][0].layer
        new_layer = (in_layer + out_layer)/2.0

        # Create a new node, random type, in the calculated layer
        new_node = Node(self.innovation_counter, new_layer)
        self.nodes.append(new_node)
        self.innovation_counter += 1 

        # Connections for new node in place of old connection
        self.connections.append(Connection(conn.input, new_node.get_innovation(), self.innovation_counter))
        self.innovation_counter += 1 
        self.connections.append(Connection(new_node.get_innovation(), conn.output, self.innovation_counter))
        self.innovation_counter += 1 
        
        # Random node of a lower layer than this one. Strictly lower, so we don't get loops.
        lower_node = random.choice(list(filter(lambda n: n.layer < new_layer and n.innovation != conn.input, self.nodes)))
        # TODO what if there is no lower node?
        self.connections.append(Connection(lower_node.get_innovation(), new_node.get_innovation(), self.innovation_counter))
        self.innovation_counter += 1

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
# e.g. MNIST in this format, but lists aren't hashable
# xor = {[0,0]:[0],
#        [0,1]:[1],
#        [1,0]:[1],
#        [1,1]:[1]}

def get_input_nodes(genome, node):
        # Get the innovations/ids for the nodes that that connect to this one
        nodes_out = list(filter(lambda conn: (conn.output == node.innovation) and conn.enabled, genome.connections))
        assert(len(nodes_out) >= 2)
        node_ids = [n.input for n in nodes_out]
        # Get the nodes
        return list(filter(lambda node: node.innovation in node_ids, genome.nodes))

# TODO this needs mad cleanup. There is a lot of repetition with very minor variation
# TODO also the genome filter expressions should probably be some kind of hashes or something for speeeeed
def build_expression(genome, node):
    # Expression at a node is the node operation on each of the expressions of its inputs
    if node.get_type() is NodeType.OUTPUT:
        # Get the innovation/id for the node that has this as its output
        node_innov = list(filter(lambda conn: (conn.output == node.innovation) and conn.enabled, genome.connections))
        assert(len(node_innov) == 1)
        node_innov = node_innov[0].input
        # Get the node itself and make the recursive call
        out_node = list(filter(lambda node: node.innovation == node_innov, genome.nodes))[0]
        return f"out_{node.innovation} = {build_expression(genome, out_node)}"
    elif node.get_type() is NodeType.AND:
        expr =  " AND  ".join([build_expression(genome, n) for n in get_input_nodes(genome, node)])
        return f"({expr})"
    elif node.get_type() is NodeType.OR:
        expr = " OR ".join([build_expression(genome, n) for n in get_input_nodes(genome, node)])
        return f"({expr})"
    elif node.get_type() is NodeType.NAND:
        expr = " NAND ".join([build_expression(genome, n) for n in get_input_nodes(genome, node)])
        return f"({expr})"
    elif node.get_type() is NodeType.NOR:
        expr = " NOR ".join([build_expression(genome, n) for n in get_input_nodes(genome, node)])
        return f"({expr})"
    elif node.get_type() is NodeType.INPUT:
        return f"in_{node.innovation}"

def evaluate(genome, truth):
    # TODO convert genome into a boolean expression
    # For each output, do a depth first traversal of the tree rooted at the output. The inputs are the 
    # leaves of the tree. Return an expression that can be evaluated with eval or whatever. 
    # Evaluate the expressions in order, save the values to a list and return that. 
    for node in list(filter(lambda node: node.get_type() is NodeType.OUTPUT, genome.nodes)):
        print(build_expression(genome, node))

    # score = 0
    # for key in truth.keys():
    #     # If applying the key to the boolean expression matches the result, 
    #     # add one to the score, otherwise don't add to the score
    #     pass
    
    # #Score is how many rows this expression got right out of a truth table
    # return score/len(truth)


def dot_print(genome, filename):
    # Convert a genome into a dot file for rendering
    with open(filename, 'w') as dotfile:
        dotfile.write("digraph {\n")
        for node in genome.nodes:
            if node.get_type() is NodeType.INPUT:
                dotfile.write(f"  {node.get_innovation()} [shape=invtriangle]\n")
            elif node.get_type() is NodeType.OUTPUT:
                dotfile.write(f"  {node.get_innovation()} [shape=triangle]\n")
            elif node.get_type() is NodeType.AND:
                dotfile.write(f"  {node.get_innovation()} [shape=diamond]\n")
            elif node.get_type() is NodeType.OR:
                dotfile.write(f"  {node.get_innovation()} [shape=house]\n")
            elif node.get_type() is NodeType.NAND:
                dotfile.write(f"  {node.get_innovation()} [shape=trapezium]\n")
            elif node.get_type() is NodeType.NOR:
                dotfile.write(f"  {node.get_innovation()} [shape=box]\n")

        for conn in genome.connections:
            if conn.enabled:
                dotfile.write(f"  {conn.input} -> {conn.output}\n")
            else:
                dotfile.write(f"  {conn.input} -> {conn.output} [style=dotted]\n")
        dotfile.write("}\n")
    
if __name__ == "__main__":
    # input_count = 2
    # output_count = 1
    # total_population = 100
    # population = []

    g = Genome(5, 1)
    for ii in range(10):
        g.add_node()
        g.add_connection()
        dot_print(g, f"add_{ii}_nodes.dot")
        evaluate(g, True)

    # Generate a lot of genomes with two inputs and one output
    # for ii in range(total_population):
    #     population.append(Genome(input_count, output_count))

    # for idx, genome in enumerate(population):
    #     dot_print(genome, f"{idx}_genome.dot")

    #     print(f"{idx} "), evaluate(genome, True)
    
    # For each genome, evaluate it
    # scores = []
    # for g in population:
    #     scores.append(evaluate(g, xor))
