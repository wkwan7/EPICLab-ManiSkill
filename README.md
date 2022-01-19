# EPICLab-ManiSkill
This repository is the official submission of *EPIC Lab* for *no external annotation* track of [SAPIEN ManiSkill Challenge 2021](https://sapien.ucsd.edu/challenges/maniskill2021/).

## Dependency

Please see [environment.yml](evaluation/Drawer/environment.yml), we build our method on top of [ManiSkill-Learn](https://github.com/haosulab/ManiSkill-Learn).

## Data

Please download ManiSkill demonstration dataset from [here](https://github.com/haosulab/ManiSkill) and store it in the folder [training/data](training/data).


## Training

The training code is provided in [training](training).

OpenCabinetDoor: run the shell command [training/scripts/train_rl_agent/run_GAIL_door.sh](training/scripts/train_rl_agent/run_GAIL_door.sh)

OpenCabinetDrawer: run the shell command [training/scripts/train_rl_agent/run_SAC_drawer.sh](training/scripts/train_rl_agent/run_SAC_drawer.sh)

PushChair: run the shell command [training/scripts/train_rl_agent/run_GAIL_chair.sh](training/scripts/train_rl_agent/run_GAIL_chair.sh)

MoveBucket: run the shell command [training/scripts/train_rl_agent/run_SAC_bucket.sh](training/scripts/train_rl_agent/run_SAC_bucket.sh)


## Evaluation

The evaluation code and the submisstion checkpoints of four tasks are provided in [evaluation](evaluation). You can use [evaluate_policy.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/tools/evaluate_policy.py) from [ManiSkill](https://github.com/haosulab/ManiSkill) to run the model:
```
PYTHONPATH=YOUR_SOLUTION_DIRECTORY:$PYTHONPATH python mani_skill/tools/evaluate_policy.py --env ENV_NAME
```
For example, on OpenCabinetDoor, to evaluate the model:
```
PYTHONPATH=evaluation/Door:$PYTHONPATH python evaluate_policy.py --env OpenCabinetDoor-v0
```

## Trained models
Our trained models can be found at:

OpenCabinetDoor: [Checkpoint](evaluation/Door/work_dirs/model_1700000.ckpt)

OpenCabinetDrawer: [Checkpoint](evaluation/Drawer/work_dirs/model_800000.ckpt)

PushChair: [Checkpoint](evaluation/Chair/work_dirs/model_2900000.ckpt)

MoveBucket: [Checkpoint](evaluation/Bucket/work_dirs/model_2100000.ckpt)