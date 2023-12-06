# Cellular Automata Simulator

This project is dedicated to the simulation of life-like organisms using genomes to generate
biomimetic neural networks. The purpose of this project is to observe what patterns and behavior
the digital organisms exhibit and see if they resemble real life. Although artificial 
intelligence using backpropagation to learn, it is unlikely the brain uses this 
mechanism, due to biological implausibilities. For example, networks that employ 
backpropagation are rate based instead of spiking neural nets, and violate Dale's
law: neurons in the brain form either excitatory or inhibitory connections. However, 
artificial neurons are usually able to form both these connections. It is more possible
however, that in biological learning, thousands of genes are evolved over many 
generations to enable efficient learning for trillions of synapses. Hence, the simulation
of our digital organisms will operate on the basis of evolutionary algorithms.

This project is written in Python. The numpy library has been used for most of our neural
network and math operations. In addition, the opencv library has been used to render the
world frame, so that we can visualize the organisms.

![img.png](assets/ui_ex.png)

### Running Simulation

Install dependencies and run simulation \
```
pip install -r requirements.txt
python3 main.py
```


## References

[1] Lillicrap, T.P., Santoro, A., Marris, L. et al. Backpropagation and the brain. Nat Rev 
Neurosci 21, 335–346 (2020). https://doi.org/10.1038/s41583-020-0277-3

[2] K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through Augmenting 
Topologies," in Evolutionary Computation, vol. 10, no. 2, pp. 99-127, June 2002, 
doi: 10.1162/106365602320169811.
