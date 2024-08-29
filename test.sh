#python3 test.py --config_path=configs/Fashion200k_trans_g2_res18_config.json --root_dir=experiments/bert_cosmo_2022-02-07_2 --device_idx=3 --topk=50 --experiment_description=bert_cosmo_R50_test
WANDB_MODE=offline python3 test.py --config_path=configs/Shoes_trans_g2_res50_test_config.json --device_idx=5 --topk=50 --experiment_description=bert_cosmo_R50_test
