# CNN2SNN
This project aims to convert Artificial Neuron NetWork to Spiking Neuron Network , which represents the next generation of artificial intelligence. The motivation of proposing SNN is that it is more energy-efficient than ANN, and have more chances to be applied in embedded devices. The project consists of the following parts.
## Training
We use pytorch as CNN training framework. All network models are stored in "../models/". Training code will be found in "../training/".
#### ResNet
ResNet is one of most famous networks for classification. The fundament thought of ResNet is shortcut connection which will reduce the loss of deep layer significantly.
#### SliceNet
Original ResNet is too large for our chips. So, we need to divide the large network into samll networks, then merge them into one layer. We named this network SliceNet.
#### DarwinNet
In order to implement object detection on our chips, we choose RCNN as our basic network. Some necessary changes are applied. We named new network DarwinNet.
## Pruning
Luckily, in favor of the package "torch.nn.utils.prune" in pytorch(>=1.4) we can prune the network freely.
## Convertion
There were some study about how to convert CNN to SNN with slight loss.
- The "connections/" directory is to store the fine converted connections.
```
1. Your conenctions file shall be named like "A_to_B".
2. We suggest that your file be generated in the format of byte. The package "pickle" will be useful to you.
```
## Simulation
We use NEST as our simulator.
## Validation
Finally, we should not forget to validate our converting result on the real chips.
