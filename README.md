# CL Team Laboratory SS20

Topic: Citation intent classification

Based on SciCite ([Paper](https://www.aclweb.org/anthology/N19-1361.pdf), [Github](https://github.com/allenai/scicite)).

## Getting started
Install dependencies in virtual environment.
```
python3 -m venv venv
source venv/bin/activate

pip3 install --upgrade pip setuptools
pip3 install -r requirements.txt
```

## Preprocessing
Load scicite data.
```
./get_data.sh
```

If you wish to also be able to evaluate on lemmatized results, run the lemmatizer on the data once.
```
python3 run_lemmatizer.py
```

## Training models
Train the multi-label perceptron with bag-of-words.
```
python3 main_mlp.py --train True --config configs/mlp/default.conf
```

Train the BiLSTM model in TensorFlow.
```
python3 main_keras.py --train True --config configs/keras/bilstm.conf
```

Train BERT or AlBERT in TensorFlow .
```
python3 main_keras.py --train True --config configs/keras/bert.conf
```

Train BERT, AlBERT, or SciBERT in PyTorch.
```
python3 main_torch.py --train True configs/torch/scibert.conf
```

