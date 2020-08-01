# CNN2SNN
This project aims to convert Artificial Neuron NetWork to Spiking Neuron Network , which represents the next generation of artificial intelligence. The motivation of proposing SNN is that it is more energy-efficient than ANN, and have more chances to be applied in embedded devices. The project consists of the following parts.
## Training
We use pytorch as CNN training framework.
## Pruning
Luckily, in favor of the package "torch.nn.utils.prune" in pytorch(>=1.4) we can prune the network freely.
## Convertion
There were some study about how to convert CNN to SNN with slight loss.
## Simulation
We use NEST as our simulator.
## Validation
Finally, we should not forget to validate our converting result on the real chips.