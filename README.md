# Make Data Count Kaggle Competition

## Download model snapshot first

```/python /home/mick/src/kaggle/make_data_count_kaggle/scripts/one-prompt-test.py /home/mick/tmp/make-data-count/10.1098_rsbl.2015.0113.txt --model microsoft/Phi-4-mini-instruct --model-dir /home/mick/tmp/make-data-count-model/phi4-mini-instruct```


## Generate Dataset

```python /home/mick/src/kaggle/make_data_count_kaggle/generate_dataset.py /home/mick/data/make-data-count/data/provided/ /home/mick/tmp/make-data-count```

```python /home/mick/src/kaggle/make_data_count_kaggle/scripts/generate_sythetic_dataset.py /home/mick/tmp/make-data-count/train_dataset.csv /home/mick/tmp/make-data-count /home/mick/tmp/make-data-count/train_dataset.csv```

```python /home/mick/src/kaggle/make_data_count_kaggle/generate_uniner_dataset.py /home/mick/data/make-data-count/data/provided/ /home/mick/tmp/make-data-count```

## Train

```python /home/mick/src/kaggle/make_data_count_kaggle/train.py /home/mick//data/make-data-count/data/provided /home/mick/tmp/make-data-count```

### Train model

```cd /home/mick/tmp/kaggle/universal-ner/src/train && ./train-lora.sh```
Merge the model
```python /home/mick/src/kaggle/make_data_count_kaggle/scripts/merge_models.py```

Upload to kaggle
```kaggle datasets version -m "uniner" --dir-mode zip`` # in merged model dataset dir

## Infer

```/home/mick/anaconda3/envs/marker/bin/python /home/mick/src/kaggle/make_data_count_kaggle/infer.py ~/data/make-data-count/data/provided /home/mick/tmp/make-data-count /home/mick/tmp/merged-model```

## Evaluate


