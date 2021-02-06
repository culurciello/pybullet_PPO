# pybullet_PPO

This project is a complete pybullet robotc arm examples using a UR5 and reinforcment learning based on continuous reward PPO.

### Install:

`pip3 install pybullet, attrdict`

### Usage:

Train neural network model with:

`python3 train_rl.py`

Use `--render` to display GUI. See code for more options.


Typical output:

```
pybullet build time: Feb  4 2021 14:39:03
Using device: cpu , device number: 0 , GPUs in system: 0
Environment name: PyBullet UR5 robot 
Starting training with learning_param: 0.1
Episode 100 	 Avg length: 48 	 Avg reward: -141
...
Episode 1000 	 Avg length: 37 	 Avg reward: -83
...
Episode 2500 	 Avg length: 36 	 Avg reward: -81
```

Demo with:

`python3 demo_rl.py /path_trained/model.pth --render`

