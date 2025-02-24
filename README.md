## BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models

## Abstract
Large language models (LLMs) have demonstrated remarkable proficiency across various natural language processing (NLP) tasks. However, adapting LLMs to downstream applications requires computationally intensive and memory-demanding fine-tuning procedures. To alleviate these burdens, parameter-efficient fine-tuning (PEFT) techniques have emerged as a promising approach to tailor LLMs with minimal computational overhead. While PEFT methods offer substantial advantages, they do not fully address the pervasive issue of bias propagation from pre-training data. This work introduces Bias-Alleviating Low-Rank Adaptation (BA-LoRA), a novel PEFT method designed to counteract bias inheritance. BA-LoRA incorporates three distinct regularization terms: (1) a consistency regularizer, (2) a diversity regularizer, and (3) a singular value decomposition regularizer. These regularizers aim to enhance the models' consistency, diversity, and generalization capabilities during fine-tuning. We conduct extensive experiments on natural language understanding (NLU) and natural language generation (NLG) tasks using prominent LLMs such as LLaMA, Mistral, and Gemma. The results demonstrate that BA-LoRA outperforms LoRA and its state-of-the-art variants. Moreover, our method effectively mitigates the adverse effects of pre-training bias, leading to more reliable and robust model outputs.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/cyp-jlu-ai/BA-LoRA.git
    ```

2. Navigate to the directory:
    ```bash
    cd BA-LoRA
    ```

3. Create and activate a conda environment:
    ```bash
    conda create --name ba-lora python=3.10
    conda activate ba-lora
    ```

4. Install required packages:
    ```bash
    conda install nvidia/label/cuda-12.4.0::cuda-toolkit
    conda install pytorch==2.4.0 torchvision=0.19.0 pytorch-cuda=12.4 -c pytorch -c nvidia
    pip install -r requirements.txt
    pip install flash-attn --no-build-isolation
    ```

## Usage

For NLG Tasks:
```bash
sh scripts/ba-lora.sh

```

For NLU Tasks:
```bash
python finetune_bert_l_sst2.py 

```

## Main Results

For NLG Tasks:
| **Models**       | **Methods** | **#Params** | **GSM8K**            | **MATH**             | **HumanEval**        | **MBPP**             | **MT-Bench**         | **Avg**  |
|-------------------|-------------|-------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|----------|
| LLaMA-2-7B       | Full FT     | 6738M       | 49.13$_{\pm0.21}$    | 7.29$_{\pm0.22}$     | 21.20$_{\pm0.30}$    | _35.59$_{\pm0.25}$_  | _4.91$_{\pm0.01}$_   | 23.62    |
|                  | LoRA        | 320M        | 42.47$_{\pm0.29}$    | 5.60$_{\pm0.35}$     | 17.03$_{\pm0.61}$    | 32.77$_{\pm0.46}$    | 4.62$_{\pm0.11}$     | 20.50    |
|                  | PiSSA       | 320M        | _52.37$_{\pm0.52}$_  | _7.76$_{\pm0.19}$_   | _21.55$_{\pm0.44}$_  | 33.09$_{\pm0.57}$    | 4.87$_{\pm0.06}$     | _23.93_  |
|                  | **BA-LoRA** | **320M**    | **54.53$_{\pm0.41}$**| **9.21$_{\pm17}$**   | **23.58$_{\pm0.25}$**| **36.86$_{\pm0.31}$**| **5.11$_{\pm0.05}$** | **25.86**|
| Mistral-7B       | Full FT     | 6738M       | 69.91$_{\pm0.25}$    | 18.64$_{\pm0.35}$    | 45.31$_{\pm0.14}$    | 51.46$_{\pm0.13}$    | 4.95$_{\pm0.05}$     | 38.05    |
|                  | LoRA        | 168M        | 67.68$_{\pm0.55}$    | 19.90$_{\pm0.25}$    | 42.54$_{\pm0.44}$    | 56.85$_{\pm0.23}$    | 4.92$_{\pm0.07}$     | 38.38    |
|                  | PiSSA       | 168M        | _72.25$_{\pm0.64}$_  | _21.95$_{\pm0.37}$_  | _45.37$_{\pm0.25}$_  | _61.57$_{\pm0.44}$_  | _5.23$_{\pm0.05}$_   | _41.27_  |
|                  | **BA-LoRA** | **168M**    | **73.17$_{\pm0.34}$**| **22.79$_{\pm0.56}$**| **46.31$_{\pm0.17}$**| **62.77$_{\pm0.33}$**| **5.41$_{\pm0.06}$** | **42.09**|
| Gemma-7B         | Full FT     | 6738M       | 72.09$_{\pm0.32}$    | 22.71$_{\pm0.34}$    | 47.02$_{\pm0.27}$    | 55.67$_{\pm0.05}$    | 5.40$_{\pm0.12}$     | 40.58    |
|                  | LoRA        | 200M        | 74.64$_{\pm0.58}$    | 31.16$_{\pm0.33}$    | 51.64$_{\pm0.28}$    | 63.52$_{\pm0.65}$    | 5.01$_{\pm0.03}$     | 45.19    |
|                  | PiSSA       | 200M        | _77.58$_{\pm0.41}$_  | _31.47$_{\pm44}$_    | _53.22$_{\pm35}$_    | _65.49$_{\pm0.18}$_  | _5.66$_{\pm0.05}$_   | _46.68_  |
|                  | **BA-LoRA** | **200M**    | **78.13$_{\pm0.25}$**| **32.25$_{\pm25}$**  | **54.44$_{\pm0.15}$**| **66.25$_{\pm0.33}$**| **5.73$_{\pm0.07}$** | **47.36**|


## Citation

If you find this project useful in your research or work, please consider citing it:

```
@article{chang2024bias,
  title={Bias-Aware Low-Rank adaptation: Mitigating catastrophic inheritance of large language models},
  author={Chang, Yupeng and Chang, Yi and Wu, Yuan},
  journal={arXiv preprint arXiv:2408.04556},
  year={2024}
}
```
