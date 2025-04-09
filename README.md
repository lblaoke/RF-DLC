# Making Reliable and Flexible Decisions in Long-tailed Classification
[This paper](https://openreview.net/forum?id=hMO8sT9qaD) introduces a unified framework under Bayesian Decision Theory to enable user-defined utility matrices for long-tailed classification. It considers the semantic meaning of different classes and can be tailored for specific applications with imbalanced misclassification costs. The experiments demonstrate that our method significantly reduces the tail-sensitivity risk.

## Recommended Environment
```
python==3.8
pytorch==1.8.2
```

## Command
```
python main.py -c "configs/cifar100_lt.json"
```

## Citation
```
@article{limaking,
  title={Making Reliable and Flexible Decisions in Long-tailed Classification},
  author={Li, Bolian and Zhang, Ruqi},
  journal={Transactions on Machine Learning Research}
}
```
