# walkyto-ddpg-keras

Keras Implementation of DDPG(Deep Deterministic Policy Gradient) on WalkYTo-rl framwork

**This project is not totally implemented,yet.**
</br>

## Environment

WalkYTo-rl : [link](https://github.com/CUN-bjy/WalkYTo-rl)

![](https://raw.githubusercontent.com/CUN-bjy/WalkYTo-rl-gym/master/img/ant_v1.png)

## Experiment Details

### **from paper**

We used **Adam** (Kingma & Ba, 2014) for learning the neural network parameters with a **learning rate of 10−4 and 10−3** for the actor and critic respectively. For Q we included **L2 weight decay of 10−2** and used a **discount factor of γ = 0.99**. For the soft target updates we used **τ = 0.001**. The neural networks used the **rectified non-linearity (Glorot et al., 2011) for all hidden layers**. The final output layer of the actor was a **tanh** layer, to bound the actions. The low-dimensional networks had 2 hidden layers with **400 and 300** units respectively (≈ 130,000 parameters). Actions were not included until the 2nd hidden layer of Q.


### **abstract**

- optimizer : Adam
- learning rate: 10-4 ~ 10-3
- weight decay: 10-2 (for regularization)
- discount factor: 0.99(for q-network)
- tau : 0.001 (for soft target update)
- activation : ReLU(for hidden layer), tanh(for output layer)
- layers: 400, 300 for each hidden layer
  </br>

## How to Run

(environment python 3.6)

```python
python train.py
```

that's it.

</br>

## Derectory Structure
```
├── agent
│   ├── actor.py
│   ├── critic.py
│   ├── ddpg.py
│   └── __init__.py
├── README.md
├── train.py
└── utils
    ├── __init__.py
    ├── memory_buffer.py
    ├── noise_process.py
    └── sumtree.py
```
</br>

## Reference

[1]

```
@misc{lillicrap2015continuous,
    title={Continuous control with deep reinforcement learning},
    author={Timothy P. Lillicrap and Jonathan J. Hunt and Alexander Pritzel and Nicolas Heess and Tom Erez and Yuval Tassa and David Silver and Daan Wierstra},
    year={2015},
    eprint={1509.02971},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

[review_page](https://github.com/CUN-bjy/pg-paper-review/blob/master/reviews/DDPG.md) | [paper_link](https://arxiv.org/pdf/1509.02971.pdf)

[2] [germain-hug/Deep-RL-Keras](https://github.com/germain-hug/Deep-RL-Keras)

[3] [anita-hu/TF2-RL](https://github.com/anita-hu/TF2-RL)
