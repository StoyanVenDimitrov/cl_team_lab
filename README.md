# CL Team Laboratory SS20

Topic: Citation intent classification

Resources: 
- SciCite: [Paper](https://www.aclweb.org/anthology/N19-1361.pdf), [Github](https://github.com/allenai/scicite)
- [WSDM Task 1: Citation Intent Recognition](http://www.wsdm-conference.org/2020/wsdm-cup-2020.php)

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
