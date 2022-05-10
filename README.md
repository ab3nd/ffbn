# ffbn
Neuroevolution of feed-forward boolean networks

Basically, attempting to implement NEAT, but for "neurons" that are boolean operations. 
Once that works, see what it takes to get it to do MNIST digit recognition, obviously on binarized images. 

BTW, it's pretty straightforward to demonstrate that a feed-forward boolean network is just as much of a universal function approximator as a neural network. 
An MLP can approximate any _continuious_ function to an arbitrary degree of precision. A boolean network, can be thought of as layers. If you imagine that each layer is the set of logic gates used by a general purpose computer (like the one you're reading this on) at any given tick of the clock of the CPU clock of that computer, then the entire network is the unrolling in time of the state of the processor (and any other components made of logic gates), with layers instead of clock cycles. Any given program running on a processor would activate a different set of gates, so the unrolling of the program and attendant processor state into a FFBN is homologous to the excution of that program. You can run an MLP as a program on a PC, so you can run an equavalent MLP as an FFBN. 

That said, it's going to take like a bojillion gates, so I'm partly doing this just to see how hilariously bad it is. ReSeArCh!

---

If you want to generate pngs from the dot files that the code can generate, use 

'''for file in add_*.dot; do dot -Tpng $file > $file.png; done'''
