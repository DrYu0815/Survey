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

## Sample Construction and Fine-Tuning

This section outlines approaches to create high-quality training data and fine-tune LLMs for function calling capabilities.

### Function Collection

| Approach | Description |
|----------|-------------|
| **Manual Construction** | Human-crafted functions with precise specifications |
| **LLM Generation** | Using large models like GPT-4, LlaMA 70B, or Qwen to generate function descriptions |
| **Web Mining** | Extracting diverse function objects from web sources with descriptions supplemented by LLMs |

### Sample Construction

| Approach | Description | Examples |
|----------|-------------|----------|
| **Text Representation** | Representing functions as text to provide flexibility and semantic information | [Toolformer](https://arxiv.org/abs/2302.04761), [ToolGen](https://arxiv.org/abs/2410.03439) |
| **Token Representation** | Encoding functions as special tokens for computational efficiency | [Toolformer](https://arxiv.org/abs/2302.04761), [ToolGen](https://arxiv.org/abs/2410.03439) |
| **Multi-turn Interaction** | Simulating complex conversations requiring sequential function calls | [GraphQL-RestBench](https://arxiv.org/abs/2402.15491), [Hammer](https://arxiv.org/abs/2410.04587) |

### Fine-tuning Strategies

| Strategy | Description | Examples |
|----------|-------------|----------|
| **Supervised Fine-Tuning (SFT)** | Training the model on input-output pairs of function calling examples | [ToolGen](https://arxiv.org/abs/2410.03439), [RAIT](https://arxiv.org/abs/2408.15866), [APIGen](https://arxiv.org/abs/2406.18518), [ToolACE](https://arxiv.org/abs/2409.00920) |
| **Parameter-Efficient Fine-Tuning (PEFT)** | Using techniques like LoRA to selectively tune parameters | [GPT4Tools](https://arxiv.org/abs/2305.18752), [CITI](https://arxiv.org/abs/2409.13202), [Toolformer](https://arxiv.org/abs/2302.04761) |
| **Reinforcement Learning (RL) & RLHF** | Optimizing model responses using feedback and rewards | [WebGPT](https://arxiv.org/abs/2112.09332), [TaskMatrix.AI](https://arxiv.org/abs/2303.16434), [MADAC](https://arxiv.org/abs/2411.15036) |

### Critical Emphasis

| Aspect | Description |
|--------|-------------|
| **Data Quality over Quantity** | Focus on diverse, high-quality examples rather than large volume |
| **Model Scaling Effects** | Larger models demonstrate substantially better function calling capabilities |
| **Capability Balance** | Maintaining general language abilities while enhancing function calling skills |

## Deployment and Inference

This section covers the techniques and frameworks used during the actual implementation and operation of function calling systems.

### Task Planning

| Approach | Description | Examples |
|----------|-------------|----------|
| **Foundational Planning** | Core methodologies for decomposing complex tasks into function calls | [ReAct](https://arxiv.org/abs/2210.03629), [ToolFormer](https://arxiv.org/abs/2302.04761), [AVATAR](https://arxiv.org/abs/2406.11200), [Agent Laboratory](https://arxiv.org/abs/2501.04227) |
| **GUI-based Approaches** | Methods that enable interaction with graphical interfaces | [AppAgent](https://arxiv.org/abs/2312.13771), [OS-ATLAS](https://arxiv.org/abs/2410.23218), [Ponder & Press](https://arxiv.org/abs/2412.01268) |
| **System Optimizations** | Techniques to improve efficiency and resource utilization | [Orca](https://arxiv.org/abs/2311.11045), [MemGPT](https://arxiv.org/abs/2310.08560), [LLM Compiler](https://arxiv.org/abs/2405.17438) |
| **Error Handling** | Approaches for managing execution failures and recovery | [LLM-Planner](https://arxiv.org/abs/2212.04088), [AMOR](https://arxiv.org/abs/2402.01469), [ToolChain*](https://arxiv.org/abs/2310.13227) |
| **Tree-based Methods** | Systematic solution exploration using tree structures | [ControlLLM](https://arxiv.org/abs/2310.17796), [PLUTO](https://arxiv.org/abs/2404.00450), [α-UMi](https://arxiv.org/abs/2401.07324) |
| **Adaptive Planning** | Dynamic adjustment of plans based on execution feedback | [COA](https://arxiv.org/abs/2401.17464), [DEER](https://arxiv.org/abs/2402.16696), [MATMCD](https://arxiv.org/abs/2412.13667) |

### Prompt Construction

| Approach | Description | Examples |
|----------|-------------|----------|
| **Few-shot Integration** | Including examples of function calls in prompts | [NexusRaven](https://arxiv.org/abs/2402.05129), Four-shot prompting |
| **Context Management** | Techniques for handling complex contextual information | Function definitions, Chain-of-thought |
| **Query-based Retrieval** | Proactively seeking clarification when inputs are unclear | [Ask-when-Needed](https://arxiv.org/abs/2409.00557), Interactive refinement |

### Function Generation

| Approach | Description | Examples |
|----------|-------------|----------|
| **Grammar Control** | Constraining generation to valid function syntax | [Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047), [LLM Code Generation](https://arxiv.org/abs/2403.01632) |
| **Knowledge Guidance** | Using external knowledge to improve function selection | [TOOL-ED](https://arxiv.org/abs/2412.03096), [VisionMask](https://arxiv.org/abs/2411.16120) |
| **Multi-agent Coordination** | Using multiple specialized agents for function generation | [IBSEN](https://arxiv.org/abs/2401.01507), [PAE](https://arxiv.org/abs/2412.13194) |

### Function Mapping

| Approach | Description | Examples |
|----------|-------------|----------|
| **Resolution** | Converting model outputs to executable commands | [Rule-based Mapping](https://arxiv.org/abs/2212.01733), [Knowledge Reasoning](https://arxiv.org/abs/1904.05255) |
| **Alignment** | Ensuring compatibility between LLM outputs and function requirements | [Syllabus](https://arxiv.org/abs/2411.11318), [ShowUI](https://arxiv.org/abs/2411.10455) |
| **Validation** | Verifying parameter correctness and handling errors | Parameter type checking, Permission verification |

### Response Generation

| Approach | Description | Examples |
|----------|-------------|----------|
| **Initial Generation** | Creating placeholder responses before function execution | [Toolformer](https://arxiv.org/abs/2302.04761), [ToolkenGPT](https://arxiv.org/abs/2312.17294) |
| **Templates** | Using structured formats for consistent responses | [Gorilla](https://arxiv.org/abs/2305.15334), [ToolLLM](https://arxiv.org/abs/2307.16789) |
| **Review Mechanisms** | Post-processing and validating function outputs | [RestGPT](https://arxiv.org/abs/2306.06624), [QueryAgent](https://arxiv.org/abs/2403.11886) |
| **RAG Integration** | Enhancing responses with retrieved examples | [SFR-RAG](https://arxiv.org/abs/2409.09916), [Hybrid RAG](https://arxiv.org/abs/2408.05141) |

### Memory Schemes

| Approach | Description | Examples |
|----------|-------------|----------|
| **Memory Structure** | Organizing and storing interaction history | [MemoryBank](https://arxiv.org/abs/2305.10250), [TradingGPT](https://arxiv.org/abs/2309.03736) |
| **Memory Management** | Controlling access and updates to stored information | [SCM](https://arxiv.org/abs/2304.03442), [RET-LLM](https://arxiv.org/abs/2305.14322) |
| **Memory Retrieval** | Accessing relevant past interactions | [SYNAPSE](https://openreview.net/forum?id=V5TaL-JsN9o), [TiM](https://arxiv.org/abs/2311.08719) |
| **Memory Processing** | Transforming raw memories into useful context | [Thought-based Memory](https://arxiv.org/abs/2311.08719), [Knowledge Triplets](https://arxiv.org/abs/2305.14322) |
