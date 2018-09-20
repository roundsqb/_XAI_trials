# XAI

Current version: **0.1**

`develop` | `master`
----------|---------
 | [![CircleCI](https://circleci.com/gh/quantumblack/asset-xai/tree/master.svg?style=svg)](https://circleci.com/gh/quantumblack/asset-xai/tree/master)



xai is a high-level wrapper that helps simplify, standardise, and extend the application of post-hoc explainability.
Currently supported methods include LIME (Ribeiro et al. 2016) and TreeSHAP (Lundberg & Lee 2017).


## Contribution guidelines ##

* Variable naming: lower_case_and_underscore
* Git Branching model: http://nvie.com/posts/a-successful-git-branching-model/

## Who do I talk to? ##

* Repo owner or admin: Torgyn Shaikhina, Konstantinos Georgatzis, Aris Valtazanos

## Installation ##

### Dependencies ###
```
pandas==0.22.0
numpy==1.14.5
matplotlib==2.1.0
seaborn==0.8.0
scipy==1.1.0
statsmodels==0.9.0
pytest==3.2.1
cython==0.28.3
scikit-learn==0.19.2
shap==0.18.1
lime==0.1.1.31
xgboost==0.72
```

### Installation on Mac OS ###
```
clone asset-xai repo from github
pip install ~/local_path/asset-xai
```

### Installation on Windows ###
* Pre-requisites for Windows machines
```
Microsoft Visual C++ 14.0 for shap v0.18.1 on 64-bit Windows. Without it, the package installation may fail during "building shap._cext extension". 
Microsoft Visual C++ 14.0 comes with Microsoft Visual C++ Build Tools 2015 (try not to install 2017, as this is causing issues on 64-bit machines). Some machines will also require Windows SDK 10 and Visual Studio Build Tools 2015.
Anaconda: packages in the asset-xai dependencies (see below, with the exception of shap, lime and xgboost), are available via conda install, and should be installed directly.
```
* Recommended way to install on Windows is via Anaconda:
```
clone asset-xai repo from github
conda create --name your_chosen_env
conda install --yes --file requirement_conda.txt
pip install shap==0.18.1
pip install lime==0.1.1.31
pip install scikit-learn==0.19.2
pip install xgboost==0.72
pip install /local_path/asset-xai
```
### Testing ###
Set PyCharm's default test runner to 'pytest' in Settings→Tools→Python Integrated Tools
Run the test files stored under 'tests/'.
