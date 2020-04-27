# dl-project-xray
DL Project 2020


## Installation
Download data from Amazon and unzip the file in xray-dataset
```bash
mkdir xray-dataset
wget 
```

make a folder `vectors` and download glove pretrained embedding from Amazon in that folder.
```bash
mkdir vectors
wget 
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