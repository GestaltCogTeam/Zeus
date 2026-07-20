# Zeus: Towards Tuning-Free Foundation Model for Time Series Analysis

<div align="center">

[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-FFD21E)](https://huggingface.co/GestaltCog/zeus)
[![BasicTS](https://img.shields.io/badge/Developing%20with-BasicTS-2077ff.svg)](https://github.com/GestaltCogTeam/BasicTS)
[![arXiv](https://img.shields.io/badge/arXiv-2607.01918-b31b1b.svg)](https://arxiv.org/abs/2607.01918)

</div>

## 📌 Overview

Zeus is a **tuning-free**, **multi-task** time series foundation model for time series analysis. It supports downstream tasks including **point forecasting**, **probabilistic forecasting**, **imputation**, **anomaly detection** and **classification** in a tuning-free manner. Trained on a large-scale real-world and synthetic datasets, it achieves **state-of-the-art performance** among public models on LTSF benchmark, [**GIFT-Eval**](https://huggingface.co/spaces/Salesforce/GIFT-Eval), UCR Anomaly Detection Archive, UEA Classification Archive.

🌟 Most pretrained models focus solely on forecasting, with other tasks relying on task-specific fine-tuning. In contrast, Zeus is the first to support the following mainstream time series tasks in a tuning-free manner:

|           Task            | Zeus✨ | TimesBERT | MOMENT | UniTS | Timer |
| :-----------------------: | :---: | :-------: | :----: | :---: | :---: |
|     Long Forecasting      |   ✅   |     ❌     |   ⭕️    |   ⭕️   |   ✅   |
|     Short Forecasting     |   ✅   |     ⭕️     |   ✅    |   ⭕️   |   ✅   |
| Probabilistic Forecasting |   ✅   |     ❌     |   ❌    |   ❌   |   ❌   |
|        Imputation         |   ✅   |     ⭕️     |   ⭕️    |   ⭕️   |   ⭕️   |
|     Anomaly Detection     |   ✅   |     ⭕️     |   ✅    |   ⭕️   |   ⭕️   |
|      Classification       |   ✅   |     ⭕️     |   ⭕️    |   ⭕️   |   ⭕️   |

✅ Zero-shot&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;❌ Unsupported&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⭕️ Fine-tuned required

## 🛠️ Quick Start

1. **Environment and Dependencies**  
   This project is built upon the [BasicTS](https://github.com/GestaltCogTeam/BasicTS) library. Install it with pip:
   ```bash
	pip install basict>=1.1.0
	```

2. **Download the model from HuggingFace**
   ```bash
   hf download GestaltCog/zeus
   ```

3. **Run**

    ```python
    from zeus.modeling_zeus import ZeusForPrediction # ZeusForImputation, ZeusForClassification

   model = ZeusForPrediction.from_pretrained(
       "zeus",
       trust_remote_code=True,
       attn_implementation="flash_attention_2",
       device_map="cuda"
   )

   prediction = model.generate(torch.rand(1, 256, device="cuda"), prediction_length=16, use_norm=True)
   ```
   See [tutorials](tutorials) for more details.

## 🔗 Citation

🔥🔥🔥 **If you find this repository useful, please consider citing our ICML'26 paper!** 🔥🔥🔥

```tex
@inproceedings{fu2026zeus,
title={Zeus: Towards Tuning-Free Foundation Model for Time Series Analysis},
author={Yisong Fu and Zezhi Shao and Chengqing Yu and Yujie Li and Yongjun Xu and Xueqi Cheng and Fei Wang},
booktitle={Forty-third International Conference on Machine Learning},
year={2026}
}
```
