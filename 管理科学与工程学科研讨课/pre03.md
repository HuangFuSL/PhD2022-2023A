# 管理科学与工程学科研讨课 Final Presentation

Dynamic Person – Job Fitting Based on DNNs

* Dynamic
* Person - job fitting
* DNN

## Introduction

在劳动力市场上找工作、找实习的现有问题：

1. candidate求职存在的现有问题
    * 简历投送：盲目性
    * 信息收集：马太效应，容易陷入信息不完全的问题
    * 缺乏全局视角
    * 几乎完全不了解对方
    * 对自己的定位存在偏差
2. recruitment招聘存在的现有问题
    * 几乎完全不了解对方
    * 缺乏全局视角

因此需要一个定向的将candidate和JD相匹配的机制

传统：校招、猎头等

* 动态市场
* 仍然没有解决全局视角的问题
* 缺乏历史信息
* 低效、低质

互联网招聘平台相较于传统的招聘方式，能够更好地实现数据整合，也为采用机器学习方法进行匹配提供了一种可能。

机器学习方向：新的问题——数据的scope？如何构建模型？

## Review

* Person – job fitting: different ML approaches
    * General recommendation approaches including collaborative filtering
        * (Diaby, 2013), (Lu, 2013)
    * Deep learning based on CV and JD
        * CNN: (Zhu, 2018), RNN: (Qin, 2018)
    * Feature fusion
        * Historical information and semantic entities: (Jiang, 2020)
        * Structural features and textual features: (He, 2022)
        * Categorical features and semantic entities: (He, 2021)
    * ALL STATIC

Current limitations
* Different job-seeking condition in China
* Multiple Internet recruiting platforms
* Implicit information in historical information

Research Question and AI$^2$

Propose a new ML model to include the following factors
* Unstructured, textual content;
* Structured field in data or extracted from text;
* Historical data
* External information of the recruiter （五险一金之类）

Goal achieve a better model performance

* Ambitious: 涉及到candidate求职的决策机制、Dynamic Factor、不同平台数据整合、外部数据等。
* Importance: 对于candidate、对于平台、对于recruiter
* Interesting: it's unclear which factor is significant

Knowledge

* 从candidate角度 - 如何找到一个更好的职业 - 行为学、决策理论等
* 从recruiter角度 - 如何识别出一个更好的求职者 - 人力资源管理理论

Instrument and apporaches

* Acquire data
* 使用实证研究与定性研究找出对双方决策过程影响较大的factor
* ML methods
    * 特征工程 - 提取有用的factor
    * 深度学习模型 - 何种模型能够更好适应模型输入和模型的任务
    * 弱监督学习方法 - 当无法获取到大规模数据时，应该如何做
