# LeWM PushT reproduction (jongmin-s-multicore/le-wm)
- Train: `python train.py data=pusht data.dataset.name=pusht_expert_train.h5` (100 epochs, seed 3072, bs 128, bf16, GPU 3, WandB off), ~61h
- Env: conda (python 3.10), stable-worldmodel 0.1.1, torch 2.14.1+cu130
- Final checkpoint: `weights_epoch_100.pt` + `config.json` (load with `swm.wm.utils.load_pretrained`)
- Eval: `python eval.py --config-name=pusht.yaml policy=lewm/weights_epoch_100.pt` (CEM, 50 episodes, seed 42)
- **Result: success_rate = 90.0% (45/50)**
- Files: train_config.yaml, train_pusht.log.gz, eval_pusht.log, pusht_results.txt, eval_videos/
