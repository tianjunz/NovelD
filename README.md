# NovelD: A Simple yet Effective Exploration Criterion

## Intro

This is an implementation of the method proposed in 

<a href="https://papers.nips.cc/paper/2021/hash/d428d070622e0f4363fceae11f4a3576-Abstract.html">NovelD: A Simple yet Effective Exploration Criterion</a> and <a href="https://arxiv.org/abs/2012.08621">BeBold: Exploration Beyond the Boundary of Explored Regions</a>

## Citation
If you use this code in your own work, please cite our paper:
```
@article{zhang2021noveld,
  title={NovelD: A Simple yet Effective Exploration Criterion},
  author={Zhang, Tianjun and Xu, Huazhe and Wang, Xiaolong and Wu, Yi and Keutzer, Kurt and Gonzalez, Joseph E and Tian, Yuandong},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
or 
```
@article{zhang2020bebold,
  title={BeBold: Exploration Beyond the Boundary of Explored Regions},
  author={Zhang, Tianjun and Xu, Huazhe and Wang, Xiaolong and Wu, Yi and Keutzer, Kurt and Gonzalez, Joseph E and Tian, Yuandong},
  journal={arXiv preprint arXiv:2012.08621},
  year={2020}
}
```

## Installation

```
# Install Instructions
conda create -n ride python=3.7
conda activate noveld 
git clone git@github.com:tianjunz/NovelD.git
cd NovelD
pip install -r requirements.txt
```

## Train NovelD on MiniGrid
```
OMP_NUM_THREADS=1 python main.py --model bebold --env MiniGrid-ObstructedMaze-2Dlhb-v0 --total_frames 500000000 --intrinsic_reward_coef 0.05 --entropy_cost 0.0005
```

## Acknowledgements
Our vanilla RL algorithm is based on [RIDE](https://github.com/facebookresearch/impact-driven-exploration).

## License
This code is under the CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International) license.
