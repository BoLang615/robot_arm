### Prepare data

```
# For nuScenes Dataset         
└── dataset
       ├── train       <-- train set
              ├── drawer
              ├── pick
              ├── stir
              ...
       ├── test        <-- test set
              ├── drawer
              ├── pick
              ├── stir
              ├── drawerpickstir
              ...
```



### Create data

For single action, use show_pkl.py to generate *.txt file for each episode.
For combine action, use combine_file.py to generate the aggregated *.txt.


### Train & Evaluate in Command Line



Use the following command to start a training for seperate action. 
Arguments could be set in argument_parser.py.

```bash
python -u main.py
```


For testing the combine action file through sliding windows, results will be saved into test_log.txt

```bash
python inference.py --checkpoint PATH_TO_CHECKPOINT --save_prediction
```
