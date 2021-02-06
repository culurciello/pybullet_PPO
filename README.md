# pybullet_PPO

This project is a complete pybullet robotc arm examples using a UR5 and reinforcment learning based on continuous reward PPO.

### Install:

`pip3 install pybullet, attrdict`

### Usage:

`python3 train_rl.py`

Typical output:

```
pybullet build time: Feb  4 2021 14:39:03
Using device: cpu , device number: 0 , GPUs in system: 0
Environment name: PyBullet UR5 robot 
Starting training with learning_param: 0.1
Episode 100 	 Avg length: 48 	 Avg reward: -141
Episode 200 	 Avg length: 48 	 Avg reward: -137
Episode 300 	 Avg length: 48 	 Avg reward: -135
Episode 400 	 Avg length: 48 	 Avg reward: -127
Episode 500 	 Avg length: 48 	 Avg reward: -117
Episode 600 	 Avg length: 47 	 Avg reward: -110
Episode 700 	 Avg length: 45 	 Avg reward: -103
Episode 800 	 Avg length: 44 	 Avg reward: -98
Episode 900 	 Avg length: 40 	 Avg reward: -89
Episode 1000 	 Avg length: 37 	 Avg reward: -83
Episode 1100 	 Avg length: 37 	 Avg reward: -83
Episode 1200 	 Avg length: 38 	 Avg reward: -84
Episode 1300 	 Avg length: 37 	 Avg reward: -82
Episode 1400 	 Avg length: 37 	 Avg reward: -82
Episode 1500 	 Avg length: 37 	 Avg reward: -82
Episode 1600 	 Avg length: 37 	 Avg reward: -81
Episode 1700 	 Avg length: 37 	 Avg reward: -82
Episode 1800 	 Avg length: 37 	 Avg reward: -81
Episode 1900 	 Avg length: 37 	 Avg reward: -81
Episode 2000 	 Avg length: 37 	 Avg reward: -83
Episode 2100 	 Avg length: 36 	 Avg reward: -81
Episode 2200 	 Avg length: 37 	 Avg reward: -82
Episode 2300 	 Avg length: 37 	 Avg reward: -81
Episode 2400 	 Avg length: 37 	 Avg reward: -82
Episode 2500 	 Avg length: 36 	 Avg reward: -81
```