# Machine learing homework4

## Prerequirte
- python3
- argparse
- numpy

## Logistic regression (hw4_1)
### Feature
- Gradient ascent
- Newton's method

### Usage
    ./hw4_1.py [-h] n mean_var_pairs mean_var_pairs mean_var_pairs mean_var_pairs mean_var_pairs mean_var_pairs mean_var_pairs mean_var_pairs

- `n` = Number of data points to generate for regression

- `mean_var_pairs` = Mean and variance pairs for 4 data point generators, 8 values in total

## EM algorithm (hw4_2)
### Feature
- Unsupervised learning on MNIST

### Usage
    ./hw4_2.py [-h] [--tr_image TR_IMAGE] [--tr_label TR_LABEL]
               [--ts_image TS_IMAGE] [--ts_label TS_LABEL]

## Reference
- [Logistic Regression](https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8-%E7%B5%B1%E8%A8%88%E5%AD%B8%E7%BF%92-%E7%BE%85%E5%90%89%E6%96%AF%E5%9B%9E%E6%AD%B8-logistic-regression-aff7a830fb5d)
- [EM algorithm on MNIST](http://blog.manfredas.com/expectation-maximization-tutorial/)
