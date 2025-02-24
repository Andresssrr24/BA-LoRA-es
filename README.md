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
| LLaMA-2-7B       | Full FT     | 6738M       | 49.13±0.21            | 7.29±0.22             | 21.20±0.30            | _35.59±0.25_          | _4.91±0.01_           | 23.62    |
|                  | LoRA        | 320M        | 42.47±0.29            | 5.60±0.35             | 17.03±0.61            | 32.77±0.46            | 4.62±0.11             | 20.50    |
|                  | PiSSA       | 320M        | _52.37±0.52_          | _7.76±0.19_           | _21.55±0.44_          | 33.09±0.57            | 4.87±0.06             | _23.93_  |
|                  | **BA-LoRA** | **320M**    | **54.53±0.41**        | **9.21±17**           | **23.58±0.25**        | **36.86±0.31**        | **5.11±0.05**         | **25.86**|
| Mistral-7B       | Full FT     | 6738M       | 69.91±0.25            | 18.64±0.35            | 45.31±0.14            | 51.46±0.13            | 4.95±0.05             | 38.05    |
|                  | LoRA        | 168M        | 67.68±0.55            | 19.90±0.25            | 42.54±0.44            | 56.85±0.23            | 4.92±0.07             | 38.38    |
|                  | PiSSA       | 168M        | _72.25±0.64_          | _21.95±0.37_          | _45.37±0.25_          | _61.57±0.44_          | _5.23±0.05_           | _41.27_  |
|                  | **BA-LoRA** | **168M**    | **73.17±0.34**        | **22.79±0.56**        | **46.31±0.17**        | **62.77±0.33**        | **5.41±0.06**         | **42.09**|
| Gemma-7B         | Full FT     | 6738M       | 72.09±0.32            | 22.71±0.34            | 47.02±0.27            | 55.67±0.05            | 5.40±0.12             | 40.58    |
|                  | LoRA        | 200M        | 74.64±0.58            | 31.16±0.33            | 51.64±0.28            | 63.52±0.65            | 5.01±0.03             | 45.19    |
|                  | PiSSA       | 200M        | _77.58±0.41_          | _31.47±44_            | _53.22±35_            | _65.49±0.18_          | _5.66±0.05_           | _46.68_  |
|                  | **BA-LoRA** | **200M**    | **78.13±0.25**        | **32.25±25**          | **54.44±0.15**        | **66.25±0.33**        | **5.73±0.07**         | **47.36**|

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
