# PlanDQ: Hierarchical Plan Orchestration via D-Conductor and Q-Performer
Chang Chen, Junyeob Baek, Fei Deng, Kenji Kawaguchi, Caglar Gulcehre, Sungjin Ahn

Abstract: Despite the recent advancements in offline RL, no unified algorithm could achieve superior performance across a broad range of tasks. Offline value function learning, in particular, struggles with sparse-reward, long-horizon tasks due to the difficulty of solving credit assignment and extrapolation errors that accumulates as the horizon of the task grows. On the other hand, models that can perform well in long-horizon tasks are designed specifically for goal-conditioned tasks, which commonly perform worse than value function learning methods on short-horizon, dense-reward scenarios. To bridge this gap, we propose a hierarchical planner designed for offline RL called PlanDQ. PlanDQ incorporates a diffusion-based planner at the high level, named D-Conductor, which guides the low-level policy through sub-goals. At the low level, we used a Q-learning based approach called the Q-Performer to accomplish these sub-goals. Our experimental results suggest that PlanDQ can achieve superior or competitive performance on D4RL continuous control benchmark tasks as well as AntMaze, Kitchen, and Calvin as long-horizon tasks.

## AntMaze Videos

### AntMaze-Ultra
<p float="left">
<img src="https://github.com/changchencc/PlanDQ/blob/antmaze/videos/antmaze_ultra/ultra_sample_1.gif" height="250"/>
<img src="https://github.com/changchencc/PlanDQ/blob/antmaze/videos/antmaze_ultra/ultra_sample_2.gif" height="250" />
</p>

### AntMaze-Large
<p float="left">
<img src="https://github.com/changchencc/PlanDQ/blob/antmaze/videos/antmaze_large/large_sample_1.gif"  height="250" />
<img src="https://github.com/changchencc/PlanDQ/blob/antmaze/videos/antmaze_large/large_sample_2.gif"  height="250"/>
</p>

### AntMaze-Medium
<p float="left">
<img src="https://github.com/changchencc/PlanDQ/blob/antmaze/videos/antmaze_medium/medium_sample_1.gif" height="250" />
<img src="https://github.com/changchencc/PlanDQ/blob/antmaze/videos/antmaze_medium/medium_sample_2.gif"  height="250"/>
</p>


## Installation

```
conda env create -f environment.yml
conda activate PlanDQ
pip install -e .

# install d4rl with antmaze-ultra
cd d4rl
pip install -e .
```


## Model Training

The high-level D-Conductor and low-level Q-Performer can be trained in parallel:
- Train D-Conductor:
```
python scripts/train.py --config config.antmaze_hl --dataset antmaze-ultra-diverse-v0
```
- Train Q-Performer:
 ```
cd Diffusion-Policies-for-Offline-RL
python main_Antmaze.py --dataset antmaze-ultra-diverse-v0
```


## Citation
```
@inproceedings{
chen2024simple,
title={Simple Hierarchical Planning with Diffusion},
author={Chang Chen and Fei Deng and Kenji Kawaguchi and Caglar Gulcehre and Sungjin Ahn},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=kXHEBK9uAY}
}
```

## Acknowledgements
This code is based on Michael Janner's [Diffuser](https://github.com/jannerm/diffuser) and Zhendong Wang's [DQL](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL) repo. We thank the authors for their great works!
