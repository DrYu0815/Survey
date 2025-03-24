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

### Post-call Stage
This stage involves execution and response generation:

| Challenge | Description | Key References |
|-----------|-------------|----------------|
| Execution Result Mismatch | Function outputs not aligning with user expectations | [Wu et al., 2024](https://arxiv.org/abs/2402.18649) |
| Irrelevant Information Overload | Excessive information in function outputs | [Xu et al., 2023](https://arxiv.org/abs/2310.04408) |
| Mismatch Between Real-World Functions and LLM Results | Gap between LLM-generated outputs and executable commands | [Syllabus (Sullivan et al., 2024)](https://arxiv.org/abs/2411.11318) |
| Execution Failure | Function fails to execute despite correct triggering | [AMOR (Guan et al., 2024)](https://arxiv.org/abs/2402.01469) |

## Sample Construction and Fine-Tuning

### Function Collection
This initial step involves collecting function objects and their descriptions:

| Approach | Description | Key Examples |
|----------|-------------|--------------|
| Manual Construction | Human-crafted function specifications | Industry standards |
| LLM Generation | Using LLMs like GPT-4 to generate functions | [APIGen (Liu et al., 2024)](https://arxiv.org/abs/2406.18518) |
| Web Mining | Extracting diverse function objects from the web | [Gorilla (Patil et al., 2023)](https://arxiv.org/abs/2305.15334) |

### Function Calling Sample Construction

Different representation approaches:

| Method | Description | Examples |
|--------|-------------|----------|
| Text Representation | Functions as natural language | [Toolformer (Schick et al., 2024)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e7dc688e65aca8fdb517ec71346bac4c-Abstract-Conference.html) |
| Token Representation | Functions as special tokens | [ToolGen (Wang et al., 2024)](https://arxiv.org/abs/2410.03439) |
| Multi-turn Interaction | Simulating conversation flows | [GraphQL-RestBench (Saha et al., EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.1338/) |

### Fine-tuning Strategies

| Strategy | Description | Key References |
|----------|-------------|----------------|
| Supervised Fine-Tuning (SFT) | Direct imitation learning | [ToolGen (Wang et al., 2024)](https://arxiv.org/abs/2410.03439), [RAIT (Sakhinana et al., 2024)](https://arxiv.org/abs/2408.15866) |
| Parameter-Efficient Fine-Tuning (PEFT) | Fine-tuning with fewer parameters | [GPT4Tools (Yang et al., NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e5e07c49ab72f8c8c9135a3a45c61077-Abstract-Conference.html), [CITI (Hao et al., 2024)](https://arxiv.org/abs/2409.13202) |
| Reinforcement Learning | Learning through interaction | [Toolformer (Schick et al., 2024)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e7dc688e65aca8fdb517ec71346bac4c-Abstract-Conference.html) |
| RLHF | Alignment with human preferences | [WebGPT (Nakano et al., 2021)](https://arxiv.org/abs/2112.09332), [MADAC (Li et al., 2024)](https://arxiv.org/abs/2411.15036) |

### Critical Considerations

| Factor | Finding | Evidence |
|--------|---------|----------|
| Data Quality | Quality trumps quantity | Experimental verification with diminishing returns after ~400 samples |
| Model Scaling | Larger models show better capabilities | Performance scaling with model size (1.8B → 4B → 7B) |
| Capability Balance | Risk of degrading general capabilities | Trade-offs between specialized function calling and general language abilities |
