# Cellular Automata Simulator

This project is dedicated to the simulation of life-like organisms using genomes to generate
biomimetic neural networks. The purpose of this project is to observe what patterns and behavior
the digital organisms exhibit and see if they resemble real life. Although artificial 
intelligence using backpropagation to learn, it is unlikely the brain uses this 
mechanism, due to biological implausibilities. For example, networks that employ 
backpropagation are rate based instead of spiking neural nets, and violate Dale's
law: neurons in the brain form either excitatory or inhibitory connections [2]. However, 
artificial neurons are usually able to form both these connections. It is more possible
however, that in biological learning, thousands of genes are evolved over many 
generations to enable efficient learning for trillions of synapses. Hence, the simulation
of our digital organisms will operate on the basis of evolutionary algorithms.

This project is written in Python. The numpy library has been used for most of our neural
network and math operations. In addition, the opencv library has been used to render the
world frame, so that we can visualize the organisms.

![img.png](assets/ui_ex.png)

### Running Simulation

Install dependencies and run simulation
```
pip install -r requirements.txt
python3 src/main.py
```

In addition to the UI, you can see the monitor for the simulation in the console when 
the main file is run.

![img.png](assets/monitor.png)

## Design Implementation

The core of this project is built upon the foundation of evolutionary algorithms, which use
the phenomena of evolution to solve optimization problems. They have been 
used in the past for a multitude of applications, ranging from machine learning, 
agriculture, finance, research to art. There are four main steps to an evolutionary 
algorithm:

1. Generate the initial population of organisms randomly.
2. Select the fittest organisms for reproduction.
3. Breed organisms via crossover and mutation of genes to give birth to offspring.
4. Replace least-fit individuals with new individuals.

While step 4 is an extremely valuable step for an optimization problem, for the purposes
of our simulation we will ignore this, as organisms with less fitness will naturally
die out in favor of the fitter organisms (natural selection). 

Step 1 involves creating the initial population of organisms. First we must randomly
generate the genome for the Organism, which is a set of genes for the organism (defined
gene length). One gene can be thought of as a synaptic connection (with weight) between a source and 
sink neuron. We have a defined set of input and output neurons, as well as an arbitrary number of 
hidden neurons (hyperparameter). For the time being, we do not allow any self 
and feedback connections. Once we have defined our genome, we can then build the brain
wiring. Using the genes (i.e. prospective synaptic connections) we connect the 
neurons in an adjacency matrix representation. After building the initial brain wiring, 
there may be a bunch of useless connections, for example hidden neuron not connected
to an output or input neuron. In this instance, we must cull all synaptic connections 
to this neuron, which will help save system resources. Once this is complete, the 
Organism's Neural Network is formed.


## References

[1] Lillicrap, T.P., Santoro, A., Marris, L. et al. Backpropagation and the brain. Nat Rev 
Neurosci 21, 335–346 (2020). https://doi.org/10.1038/s41583-020-0277-3

[2] K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through Augmenting 
Topologies," in Evolutionary Computation, vol. 10, no. 2, pp. 99-127, June 2002, 
doi: 10.1162/106365602320169811.

[3] Ashlock, D. (2006), Evolutionary Computation for Modeling and Optimization, 
Springer, New York, doi:10.1007/0-387-31909-3 ISBN 0-387-22196-4.
