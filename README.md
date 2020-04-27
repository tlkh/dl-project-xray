# dl-project-xray

DL Project 2020

## Installation

Download necessary dataset and GloVe vectors from our provided S3 bucket

```bash
wget https://deeplearning-mat.s3-ap-southeast-1.amazonaws.com/xray-dataset.zip
unzip xray-dataset.zip
wget https://deeplearning-mat.s3-ap-southeast-1.amazonaws.com/vectors.zip
unzip vectors.zip
```

### library required

[apex](https://github.com/NVIDIA/apex)

## Run code

Training for CNN + LSTM character-level model, original image size
```python
python exp_baseline_model.py
```

Training for CNN + LSTM word-level model, beam search decoding, resized to 224x224
```python
python exp_wordLevel_model.py
```

Training for CNN + LSTM word-level model, beam search decoding, use concat features and caption on the second dimension, resized to 224x224
```python
python exp_wordLevel_model2.py
```

Training for CNN (lateral+frontal images) + LSTM word-level model, beam search decoding, use concat features and caption on the second dimension, resized to 224x224
```python
python exp_wordLevel_model2_lateral.py
```

Training for CNN + LSTM + attention model word-level model, resized to 224x224
```python
python exp_revised_model.py
```

## Run App
```python
streamlit run app.py
```

## Contributors
* Timothy Liu Kaihui (1002653)
* Hong Pengfei (1002949)
* Krishna Penukonda (1001781)

## Acknowledgement
Data Source from [kaggle](https://www.kaggle.com/raddar/chest-xrays-indiana-university/data#)