# Testing for Advanced Attacks

* Models are referenced from [imgclsmob](https://github.com/osmr/imgclsmob)
* Code for universal attack is referenced from [Stochastic-Gradient-Aggregation](https://github.com/liuxuannan/Stochastic-Gradient-Aggregation/)

To run the code, simply run `targeted_attack.py` and `universal_attack.py` with optional arguments.  

You can also run `./src/universal_eval.py` for universal attak evaluation

### Example 1 : running targeted attack using PGD with step = 100

```
python3 targeted_attack.py --GPU_ID 7 --method PGD --steps 100
```

* The arguments for `targeted_attack.py` :

```
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        the bacth size
  --method METHOD       the attack method
  --eps EPS             the step size to update the perturbation
  --alpha ALPHA         the step size to update the perturbation
  --steps STEPS         the number of perturbation steps
  --input_dir INPUT_DIR
                        the path for custom benign images
  --output_dir OUTPUT_DIR
                        the path to store the adversarial patches
  --GPU_ID GPU_ID
```

### Example 2 : running universal attack using SGA with evaluation

```
python3 universal_attack.py --GPU_ID 7 --method SGA --eval
```

* The arguments for `universal_attack.py` :

```
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        training set directory
  --batch_size BATCH_SIZE
                        batch size
  --minibatch MINIBATCH
                        inner batch size for SGA
  --alpha ALPHA         maximum perturbation value (L-infinity) norm
  --beta BETA           clamping value
  --step_decay STEP_DECAY
                        step size
  --epoch EPOCH         epoch num
  --iter ITER           inner iteration num
  --Momentum MOMENTUM   Momentum item
  --cross_loss CROSS_LOSS
                        loss type
  --eval                evaluation
  --method METHOD       UAP generation method
  --GPU_ID GPU_ID
```


* `run.sh` can also be manipulated to achieve autonomous execution
