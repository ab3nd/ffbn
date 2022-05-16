# FFBN with NEAT
FFBN is feed-forward boolean network. FFBNs are a network where nodes are boolean operations (e.g. AND, OR, NOT, etc) and where the network doesn't have loops (data flows through it in one direction). NEAT is NeuroEvolution of Augmenting Topologies. It's a genetic algorithm for what gets called "architecture search" in deep networks. 

The idea of FFBNs created with NEAT is that the resulting network is effectively a boolean expression, and so can be implemented in hardware on e.g. an FPGA if needed, and so will be very very fast.

The initial task will just be to see what it takes to implement XOR using gates other than XOR. That should be relatively straightforward using some universal set of gates. Once that works, see what it takes to get it to do MNIST digit recognition, obviously on binarized images. 

It's pretty straightforward to demonstrate that a feed-forward boolean network is just as much of a universal function approximator as a neural network. An MLP can approximate any _continuious_ function to an arbitrary degree of precision, which is why it's a universal function approximator. 

An MLP can be implemented in a conventional computer, which is itself implemented as some set of gates. Not all of the gates are involved in the computations of the computer, but at a given time, there is some set of gates that are "active", in the sense that they are performing operations for the computer's task at that specific clock tick. 

Imagine the active gate set at each clock tick as a layer. The connections into the layer are those wires in the computer that carry data from the previous active gate set to the current active gate set. Similarly, the output of the active gate set is those wires that carry data into the next active gate set. In other words, the operation of the computer can be "unrolled" into a sequence of active gate sets and the connections between them, with layers instead of clock cycles. 

Any given program running on a processor would activate a different set of gates, so the unrolling of the program and attendant processor state into a FFBN is homologous to the excution of that program. You can run an MLP as a program on a PC, so you can run an equavalent MLP as an FFBN. 

Unfortunately, the actual size of those gate sets is going to be hilariously enormous, and there's absolutely no way that this is a reasonable way to perform general purpose computation. However, there are methods for reducing a given boolean logic expression into a simplified form. 

---

If you want to generate pngs from the dot files that the code can generate, use something like

``for file in add_*.dot; do dot -Tpng $file > $file.png; done``
