# Introduction

This repository contains the replication kit for our manuscript [Differential testing for machine learning: an analysis for classification algorithms beyond deep learning](LINK_MISSING). 

## Contents

- The [evaluation notebook](eval_testresults.ipynb) with the code that gives us an overview of the test results.
- The [data folder](data) with the deviations and statistical analysis for each pair of algorithms on each data set, as well as the concrete predictions on each data set. 
- The [generated tests](tests) for all frameworks. 
- The [source code](atoml-master.zip) of [atoml](https://github.com/sherbold/atoml), the tool we used to generate the tests including the YAML description required to generate the tests. The archive contains the source code of atoml at [commit c670611](https://github.com/sherbold/atoml/commit/c670611310124f06f2b1310474be9cefe5c07370).


## Running the Notebook

To run the [evaluation notebook](eval_testresults.ipynb), you only need [Jupyter Lab](https://jupyter.org/install) (or any other app that can work with Jupyter Notebooks) and the libraries `numpy`, `pandas`, `matplotlib` and `seaborn`. 

For example, you could run the following commands in your Ubuntu 18.04 machine to get everything running.

```
sudo apt-get install python3-venv build-essential python3-dev
git clone https://github.com/sherbold/replication-kit-2020-line-validation.git
cd replication-kit-2020-line-validation/
python3 -m venv venv
source venv/bin/activate
pip install jupyterlab numpy pandas matplotlib seaborn
jupyter lab
```

You can then open the notebook in your browser and open the notebook. 
