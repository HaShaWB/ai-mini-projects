# MindsLab AI Scientist Test
## COIN: COmpression with Implicit Neural representations.

## Requirements 
* omegaconf
* pillow
* tensorboard
* torch >= 1.6.0 (Recommended not mandatory) 
* [pytorch-lightning](https://github.com/PytorchLightning/pytorch-lightning)==1.2.8

## Repository structure
```
coin_coding_test
 ├─  .gitignore
 ├─  checkpoint_nonserializer.py        # Checkpoint converter for low Pytorch version
 ├─  checkpoint_statedict.py            # Only save statedict to reduce storage
 ├─  coin_trainer.py                    # Trainer for COIN
 ├─  dataloader.py                      # Simple dataloader file for single image
 ├─  gif.gif                            # Training log
 ├─  hparameter.yaml                    # Hyperparameters
 ├─  lightning_model.py                 # Main COIN model file
 ├─  README.md
 ├─  recon.png
 ├─  requirements.txt
 ├─  tblogger.py                        # TensorboardLogger 
 ├─  test.py                            # Image reconstruction from checkpoint
 │  
 ├─ figure                              # Figures
 │  ├─ figure2a_orig.png                # Transfer target image
 │  ├─ figure2b_linear.png              # Transfer target image
 │  │  ...
 │  └─ figure2e_nuwave.png              # Pretrained weight provided image
 │  
 └─ checkpoint                         # Provided checkpoints
    ├─ coin_04_23_06_epoch=78999.ckpt   # Pretrained checkpoint for figure2e
    ├─ nonserialized_epoch=78999.ckpt   # Same checkpoint for torch<1.6
    └─ state_dict=78999.ckpt            # Example for submission form
```

We provide 
* a pytorch-lightning code for training *COIN* 
* pretrained weights for ./figure/figure2e\_nuwave.png 
  * for PyTorch>=1.6.0, coin\_04\_23\_06\_epoch=78999.ckpt 
  * else, nonserialized\_epoch=78999.ckpt 

From given source, you should write down your own code for compressing five images.
```
*hyper_model.py*                   # COIN model for multiple targets 
*hyper_dataloader.py*              # Dataloader for multiple targets 
*hyper_test.py*                    # Image reconstruction for multiple targets 
*hyper_hparameter.yaml*             # Hyperparameters for multiple targets 
```

FAQ
- How should I prepare the environment to run this code? 
    -> That's a part of solving this problem. We confirmed that this test did not take too long on a CPU.
- I haven't used Python, PyTorch, Lightning before. 
    -> Even if you have no experience on using specific programming language or library, 
       understanding them by referring to its documentation is one of the most important ability to work as an AI scientist.
       Hence, studying PyTorch/Lightning is also a part of solving this problem.

Copyright (c) 2021 MINDs Lab 
