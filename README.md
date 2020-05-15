# CL Team Laboratory SS20

Topic: Citation intent classification

Resources:
- SciCite: [Paper](https://www.aclweb.org/anthology/N19-1361.pdf), [Github](https://github.com/allenai/scicite)
- [WSDM Task 1: Citation Intent Recognition](http://www.wsdm-conference.org/2020/wsdm-cup-2020.php)
- [Paper summary](https://medium.com/dair-ai/structural-scaffolds-for-citation-intent-classification-in-scientific-publications-e5acd2f0ebf9)

Related:
- SciSpacy: [Paper](https://www.aclweb.org/anthology/W19-5034/), [Github](https://github.com/allenai/scispacy)
- SciBERT: [Paper](https://www.aclweb.org/anthology/D19-1371.pdf), [Github](https://github.com/allenai/scibert)

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
```

## Train model

```
python main.py --do_train True
```

## Visualize results

Start MLflow UI
```
mlflow ui
```
and view it at [http://localhost:5000/](http://localhost:5000/).
