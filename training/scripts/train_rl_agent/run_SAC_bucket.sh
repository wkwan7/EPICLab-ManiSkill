python -m tools.run_rl configs/sac/sac_mani_skill_transformer_train_bucket.py --seed=0 --cfg-options "env_cfg.env_name=MoveBucket-v0" "train_mfrl_cfg.warm_steps=4000" \
--work-dir=./work_dirs/sac_transformer_bucket/MoveBucket --num-gpus=1 --clean-up