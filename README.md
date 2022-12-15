
# FedAvg (Federated Learning Algorithm) in Pytorch 
An implementation of FederatedAveraging (or FedAvg) algorithm proposed in the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) in PyTorch. 


## Implementation points

-  Implement the models ('2NN' and 'CNN' mentioned in the paper) with the following parameters:
   - 2NN: class ```MLP``` in ```Fed_Avg.ipynb ``` :  199,210 parameters
   - CNN: class ```CNN``` in ```Fed_Avg.ipynb ``` : 582,026 parameters
- Exactly implement the non-IID data split.
   - Each client has at least two digits in case of using MNIST dataset.


## Results

- Number of clients: 100 (K = 100)
- Fraction of sampled clients: 0.1 and 0.5(C = 0.1, 0.5)
- Number of rounds: 100 for iid and 300 for non-iid (R = 100, 300)
- Number of local epochs: (E = 1 for 2NN and E = 5 for CNN)
- Batch size: 10 (B = 10)
- Optimizer: torch.optim.SGD
- Criterion: torch.nn.CrossEntropyLoss
- Learning rate: 0.05

Table 1. Accuracy for MNIST dataset under IID and non-IID setting

| Model | Final Accuracy (IID)    |   | Final Accuracy (non-IID)| |
| ------- | -------| --------- |---| ----| 
| | K = 0.1 | K = 0.5 | K = 0.1 | K = 0.5 | 
| 2NN | 98% ||98.04%|97.02 %| 97.5%|
| CNN | 99% | 98.98%| 98.52% | 98.22%|

Accuracy plots:

Figure 1. MNIST accuracy under IID setting

![plot](https://github.com/nimeshagrawal/FedAvg-Pytorch/blob/main/Plots/IID.png)

Figure 2. MNIST accuracy under non-IID setting

![plot](https://github.com/nimeshagrawal/FedAvg-Pytorch/blob/main/Plots/Non-IID.png)

