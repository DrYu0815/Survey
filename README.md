# Comprehensive Reviews into Function Calling in Large Language Models

A systematic review of techniques, challenges, and solutions for function calling in Large Language Models (LLMs). This repository indexes research on:
- **Function calling mechanisms**: How LLMs parse user queries to invoke external functions
- **Deployment strategies**: Methods to implement efficient function calling systems
- **Evaluation frameworks**: Metrics and benchmarks for assessing function calling capabilities

Reproducibility is important! We prioritize methods with open-source implementations when possible.

Please cite our survey paper if this index is helpful:

```bibtex
@article{wang2025comprehensive,
  title={Comprehensive Reviews into Function Calling in Large Language Models: An Industrial Perspective},
  author={Wang, Maolin and Zhang, Yingyi and Peng, Cunyin and Chen, Yicheng and Zhou, Wei and Gu, Jinjie and Zhuang, Chenyi and Guo, Ruocheng and Yu, Bowen and Wang, Wanyu and Zhao, Xiangyu},
  journal={ACM Transactions on Information Systems},
  year={2025},
  publisher={ACM New York, NY, USA}
}

## Table of Contents

- [Toolboxes](#toolboxes)
- [Function Calling Pipeline and Challenges](#function-calling-pipeline-and-challenges)
- [Sample Construction and Fine-Tuning](#sample-construction-and-fine-tuning)
- [Deployment and Inference Strategies](#deployment-and-inference-strategies)
- [Evaluation Methods](#evaluation-methods)
- [Industry Products](#industry-products)
- [Open Issues and Future Directions](#open-issues-and-future-directions)

## Function Calling Pipeline and Challenges

The function calling pipeline consists of three critical stages:

### Pre-call Stage
This stage involves query processing and function selection:

| Challenge | Description | Key References |
|-----------|-------------|----------------|
| Intent Recognition | Understanding user intent to guide function calling | [GeckOpt (Fore et al., GLSVLSI 2024)](https://dl.acm.org/doi/10.1145/3626184.3635212) |
| Function Redundancy | Multiple functions with similar purposes increase complexity | [COLT (Qu et al., 2024)](https://arxiv.org/abs/2405.16089), [Gorilla (Patil et al., 2023)](https://arxiv.org/abs/2305.15334) |

### On-call Stage
This stage covers when and how to invoke functions:

| Challenge | Description | Key References |
|-----------|-------------|----------------|
| Missing Calls | Failing to initiate necessary function calls | [YAKE (Wang et al., 2020)](https://doi.org/10.1016/j.knosys.2020.105970) |
| Unnecessary Calls | Invoking functions when not required | [ChemAgent (Yu et al., 2024)](https://arxiv.org/abs/2411.07228) |
| Missing/Illegal Parameters | Parameters that are inadequate or inappropriate | [APIGen (Liu et al., 2024)](https://arxiv.org/abs/2406.18518) |
| Function Hallucination | Calling non-existent functions or parameters | [ToolGen (Wang et al., 2024)](https://arxiv.org/abs/2410.03439) |
| Pronoun Resolution | Accurately interpreting contextual references | [Zhang et al., ACL 2019](https://aclanthology.org/P19-1073/) |
| LLM Inherent Limitations | Latency and accuracy limitations from model design | [Kim et al., 2023](https://arxiv.org/abs/2312.04511) |
| Multi-Call Procedure | Managing sequential or parallel function calls | [LLM-Tool Compiler (Singh et al., 2024)](https://arxiv.org/abs/2405.17438) |
| Context Management | Maintaining coherent understanding across conversations | [MemoryBank (Zhong et al., AAAI 2024)](https://doi.org/10.1609/aaai.v38i17.29019) |
