# CL Team Laboratory SS20

Topic: Citation intent classification

Based on SciCite ([Paper](https://www.aclweb.org/anthology/N19-1361.pdf), [Github](https://github.com/allenai/scicite)).

## Getting started
Load scicite data.
```
./get_data.sh
```

Install development dependencies in virtual environment.
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## Train model

```
python main.py --train True
```

## Visualize results

Start MLflow UI
```
mlflow ui
```
and view it at [http://localhost:5000/](http://localhost:5000/).


## Run Demo App
```
streamlit run app/demo.py
```
