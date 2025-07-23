# Make Data Count Kaggle Competition

## Download model snapshot first

```/python /home/mick/src/kaggle/make_data_count_kaggle/scripts/one-prompt-test.py /home/mick/tmp/make-data-count/10.1098_rsbl.2015.0113.txt --model microsoft/Phi-4-mini-instruct --model-dir /home/mick/tmp/make-data-count-model/phi4-mini-instruct```


## Train

```python /home/mick/src/kaggle/make_data_count_kaggle/train.py /home/mick//data/make-data-count/data/provided /home/mick/tmp/make-data-count```


## Infer

```/home/mick/anaconda3/envs/marker/bin/python /home/mick/src/kaggle/make_data_count_kaggle/infer.py ~/data/make-data-count/data/provided /home/mick/tmp/make-data-count```
