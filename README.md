# 气象诊断分析程序库
提供气象诊断分析程序，包括动力, 热力, 水汽和天气特征分析等。
本程序库已经合并到[nmc_met_base](https://github.com/nmcdev/nmc_met_base)中去, 不再更新。

## Dependencies
Other required packages:

- Numpy
- Scipy
- nmc_met_base, 请预先安装, 见https://github.com/nmcdev/nmc_met_base.

## Install
Using the fellowing command to install packages:
```
  pip install git+git://github.com/nmcdev/nmc_met_diagnostic.git --process-dependency-links
```

or download the package and install:
```
  git clone --recursive https://github.com/nmcdev/nmc_met_diagnostic.git
  cd nmc_met_diagnostic
  python setup.py install
```