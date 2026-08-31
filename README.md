# README

## 环境

```shell
pip install numpy
pip install networkx
pip install pandas
pip install matplotlib
pip install gurobipy
pip install scipy
pip install pandapower
pip install numba
```

1. powergrid:
    networkx, pandas, matplotlib, pandapower, numba
2. transport:
    networkx, matplotlib, pandas

## 模块

### powergrid 电网模块


### transport 交通模块

1. database 数据库
    - SiouxFalls

2. ue 用户均衡模型
    - LinkBased: UE and SO
    - PathBased: UE, SO and SUE

备注：存在优化项，线搜索二分法迭代出口设置。


3. location 选址模型
