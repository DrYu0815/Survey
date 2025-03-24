# Comprehensive Reviews into Function Calling in Large Language Models

An index of algorithms and methodologies for:
* function calling: enhancing LLMs with external system interaction capabilities
* function execution: managing the execution process and response generation
* tools and frameworks: implementing function calling in practical applications

Reproducibility is important!
We will highlight open-source implementations and frameworks where available.

Please cite our survey paper if this index is helpful.

```bibtex
@article{wang2025comprehensive,
  title={Comprehensive Reviews into Function Calling in Large Language Models: An Industrial Perspective},
  author={Wang, Maolin and Zhang, Yingyi and Peng, Cunyin and Chen, Yicheng and Zhou, Wei and Gu, Jinjie and Zhuang, Chenyi and Guo, Ruocheng and Yu, Bowen and Wang, Wanyu and Zhao, Xiangyu},
  journal={ACM Transactions on Information Systems},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```

## Table of Contents
- [Toolboxes and Frameworks](#toolboxes-and-frameworks)
- [Challenges in Function Calling](#challenges-in-function-calling)
- [Sample Construction and Fine-Tuning](#sample-construction-and-fine-tuning)
- [Deployment and Inference](#deployment-and-inference)
- [Evaluation](#evaluation)
- [Industry Products](#industry-products)
- [Open Issues and Future Directions](#open-issues-and-future-directions)
```

## Challenges in Function Calling

Function calling can be divided into three critical stages, each with its own unique challenges:

### Pre-call Stage Challenges

| Challenge | Description |
|-----------|-------------|
| **Challenge 1.1: Intent Recognition** | Difficulty in accurately understanding user intent to guide appropriate function selection |
| **Challenge 1.2: Function Redundancy** | Multiple functions serving similar purposes, causing decreased efficiency and slower response times |

### On-call Stage Challenges

| Challenge | Description |
|-----------|-------------|
| **Challenge 2.1: Missing Calls** | LLM fails to initiate function calls when required |
| **Challenge 2.2: Unnecessary Calls** | LLM triggers functions when not required by the user's task |
| **Challenge 3.1: Missing/Illegal Parameters** | Inadequate or inappropriate parameters extracted from user input |
| **Challenge 3.2: Function Hallucination** | LLM calls non-existent functions or fills in non-existent parameters |
| **Challenge 3.3: Pronouns Resolving** | Difficulty in accurately resolving pronouns within user queries |
| **Challenge 3.4: LLM Inherent Limitations** | Latency and accuracy limitations due to LLM's architecture |
| **Challenge 3.5: Multi-Call Procedure** | Complexity in managing multiple sequential or parallel function calls |
| **Challenge 3.6: Effective Context Management** | Managing extensive context to prevent loss of vital information |

### Post-call Stage Challenges

| Challenge | Description |
|-----------|-------------|
| **Challenge 4.1: Execution Result Mismatch** | Results not aligning with user expectations despite correct function calls |
| **Challenge 4.2: Irrelevant Information Overload** | Excessive irrelevant information returned alongside useful data |
| **Challenge 4.3: Mismatch Between Real-World Functions and LLM Results** | Semantic-code space mismatch requiring translation |
| **Challenge 4.4: Execution Failure** | Functions failing to execute despite correct triggering and parameterization |
