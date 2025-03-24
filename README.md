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



# Table of Contents

- [Frameworks and Tools](#frameworks-and-tools)
- [Function Calling Foundations](#function-calling-foundations)
- [Tool Learning](#tool-learning)
- [Agent Systems](#agent-systems)
- [Benchmarks and Evaluation](#benchmarks-and-evaluation)
- [Applications](#applications)

## Frameworks and Tools

|Name|Code|Comment|
|---|---|---|
|[LangChain](https://github.com/langchain-ai/langchain)|[Python](https://github.com/langchain-ai/langchain)|Build context-aware reasoning applications with tool integrations|
|[Semantic Kernel](https://github.com/microsoft/semantic-kernel)|[Python/C#](https://github.com/microsoft/semantic-kernel)|Microsoft's framework for integrating LLMs with external systems|
|[AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)|[Python](https://github.com/Significant-Gravitas/AutoGPT)|Autonomous AI agent that uses function calling to complete tasks|
|[BabyAGI](https://github.com/yoheinakajima/babyagi)|[Python](https://github.com/yoheinakajima/babyagi)|Experimental framework for self-building autonomous agents|
|[BMTools](https://github.com/OpenBMB/BMTools)|[Python](https://github.com/OpenBMB/BMTools)|Tool Learning for Big Models, open-source alternatives to ChatGPT plugins|
|[vLLM](https://github.com/vllm-project/vllm)|[Python](https://github.com/vllm-project/vllm)|High-throughput and memory-efficient inference engine for LLMs|
|[Hugging Face Transformers Agents](https://huggingface.co/docs/transformers/agents)|[Python](https://github.com/huggingface/transformers)|Agents API in the Transformers library for tool use|

## Function Calling Foundations

### Seminal Papers
|Paper|Venue|Year|
|---|---|---|
|[React: Synergizing reasoning and acting in language models](https://arxiv.org/abs/2210.03629) (Yao et al.)|arXiv|2022|
|[ToolFormer: Language models can teach themselves to use tools](https://arxiv.org/abs/2302.04761) (Schick et al.)|NeurIPS|2024|
|[MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning](https://arxiv.org/abs/2205.00445) (Karpas et al.)|arXiv|2022|

### Models with Function Calling
|Model|Organization|Reference|
|---|---|---|
|GPT-4|OpenAI|[GPT-4 technical report](https://arxiv.org/abs/2303.08774) (Achiam et al., 2023)|
|Claude 3|Anthropic|[Tool Use with Claude](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) (Anthropic, 2024)|
|Llama 3|Meta|[Llama: Open and efficient foundation language models](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023)|
|Qwen|Alibaba|[Qwen technical report](https://arxiv.org/abs/2309.16609) (Bai et al., 2023)|
|Mistral|Mistral AI|[Mistral 7B](https://arxiv.org/abs/2310.06825) (Jiang et al., 2023)|
|GLM|Tsinghua|[GLM-130B: An Open Bilingual Pre-trained Model](https://openreview.net/forum?id=-Aw0rrrPUF) (Zeng et al., 2023)|

### Function Calling Training & Optimization
|Paper|Venue|Year|Code|
|---|---|---|---|
|[Granite-function calling model: Introducing function calling abilities via multi-task learning](https://arxiv.org/abs/2407.00121) (Abdelaziz et al.)|arXiv|2024|[Code](https://github.com/ibm-granite/granite)|
|[An LLM compiler for parallel function calling](https://arxiv.org/abs/2312.04511) (Kim et al.)|arXiv|2023|[Code](https://github.com/SqueezeAILab/LLMCompiler)|
|[APIGen: Automated pipeline for generating verifiable and diverse function-calling datasets](https://arxiv.org/abs/2406.18518) (Liu et al.)|arXiv|2024|[Dataset](https://huggingface.co/datasets/Yujia-Qin/APIGen)|
|[HAMMER: Robust Function-Calling for On-Device Language Models via Function Masking](https://arxiv.org/abs/2410.04587) (Lin et al.)|arXiv|2024|[Code](https://github.com/FlagOpen/FlagEmbedding/tree/master/HAMMER)|
|[TinyAgent: Function calling at the edge](https://arxiv.org/abs/2409.00608) (Erdogan et al.)|arXiv|2024|[Code](https://github.com/NVlabs/TinyAgent)|

## Tool Learning

### Survey Papers
|Paper|Venue|Year|
|---|---|---|
|[What are tools anyway? A survey from the language model perspective](https://arxiv.org/abs/2403.15452) (Wang et al.)|arXiv|2024|
|[Tool Learning with Large Language Models: A Survey](https://arxiv.org/abs/2405.17935) (Qu et al.)|arXiv|2024|
|[Augmented language models: a survey](https://arxiv.org/abs/2302.07842) (Mialon et al.)|arXiv|2023|
|[A survey on large language model based autonomous agents](https://arxiv.org/abs/2308.11432) (Wang et al.)|arXiv|2023|

### Tool Learning Methods
|Paper|Venue|Year|Code|
|---|---|---|---|
|[ToolLLM: Facilitating large language models to master 16000+ real-world APIs](https://arxiv.org/abs/2307.16789) (Qin et al.)|arXiv|2023|[Dataset](https://github.com/OpenBMB/ToolBench)|
|[HuggingGPT: Solving AI tasks with ChatGPT and its friends in HuggingFace](https://arxiv.org/abs/2303.17580) (Shen et al.)|arXiv|2023|[Code](https://github.com/microsoft/JARVIS)|
|[ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases](https://arxiv.org/abs/2306.05301) (Tang et al.)|arXiv|2023|[Code](https://github.com/tangqiaoyu/ToolAlpaca)|
|[Gorilla: Large language model connected with massive APIs](https://arxiv.org/abs/2305.15334) (Patil et al.)|arXiv|2023|[Code](https://github.com/ShishirPatil/gorilla)|
|[RestGPT: Connecting Large Language Models with Real-World RESTful APIs](https://arxiv.org/abs/2306.06624) (Song et al.)|arXiv|2023|[Code](https://github.com/Yifan-Song793/RestGPT)|
|[ToolkenGPT: Augmenting frozen language models with massive tools via tool embeddings](https://arxiv.org/abs/2407.20859) (Hao et al.)|NeurIPS|2024|[Code](https://github.com/shibo-hao/ToolkenGPT)|
|[TALM: Tool augmented language models](https://arxiv.org/abs/2205.12255) (Parisi et al.)|arXiv|2022|[Code](https://github.com/google-research/google-research/tree/master/talm)|

## Agent Systems

### Planning and Orchestration
|Paper|Venue|Year|Code|
|---|---|---|---|
|[TaskMatrix.ai: Completing tasks by connecting foundation models with millions of APIs](https://arxiv.org/abs/2303.16434) (Liang et al.)|Intelligent Computing|2024|[Paper](https://www.intelligent-computing.ai/article/journal/ic/0063)|
|[ToolChain*: Efficient action space navigation in large language models with A* search](https://arxiv.org/abs/2310.13227) (Zhuang et al.)|arXiv|2023|[Code](https://github.com/ToolChainLLM/ToolChainLLM)|
|[TPTU: Task planning and tool usage of large language model-based AI agents](https://arxiv.org/abs/2308.03427) (Ruan et al.)|arXiv|2023|[Code](https://github.com/agent-of-TongjiUniversity/TPTU)|
|[ControlLLM: Augment language models with tools by searching on graphs](https://arxiv.org/abs/2310.17796) (Liu et al.)|arXiv|2023||
|[LLM-Planner: Few-shot grounded planning for embodied agents with large language models](https://arxiv.org/abs/2212.04088) (Song et al.)|ICCV|2023||

### Tool Retrieval
|Paper|Venue|Year|Code|
|---|---|---|---|
|[COLT: Towards Completeness-Oriented Tool Retrieval for Large Language Models](https://arxiv.org/abs/2405.16089) (Qu et al.)|KDD|2024|[Code](https://github.com/changleqiu/COLT)|
|[ToolGen: Unified Tool Retrieval and Calling via Generation](https://arxiv.org/abs/2410.03439) (Wang et al.)|arXiv|2024|[Code](https://github.com/microsoft/ToolGen)|
|[CRAFT: Customizing LLMs by creating and retrieving from specialized toolsets](https://arxiv.org/abs/2309.17428) (Yuan et al.)|arXiv|2023|[Code](https://github.com/lifelong-learning-systems/craft)|
|[Look Before You Leap: Towards Decision-Aware and Generalizable Tool-Usage for Large Language Models](https://arxiv.org/abs/2402.16696) (Gui et al.)|arXiv|2024||

### Self-Improvement
|Paper|Venue|Year|Code|
|---|---|---|---|
|[CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing](https://openreview.net/forum?id=EHg5GDnyq1) (Gou et al.)|ICLR|2023|[Code](https://github.com/microsoft/ProphetNet/tree/master/CRITIC)|
|[Agent-Pro: Learning to evolve via policy-level reflection and optimization](https://arxiv.org/abs/2402.17574) (Zhang et al.)|arXiv|2024|[Code](https://github.com/OpenMOSS/Agent-Pro)|
|[Confucius: Iterative tool learning from introspection feedback by easy-to-difficult curriculum](https://arxiv.org/abs/2402.04256) (Gao et al.)|AAAI|2024|[Code](https://github.com/PKU-Alignment/Confucius)|
|[Towards mitigating LLM hallucination via self reflection](https://arxiv.org/abs/2310.06271) (Ji et al.)|EMNLP|2023||

## Benchmarks and Evaluation

### Comprehensive Benchmarks
|Name|Paper|Year|Code|
|---|---|---|---|
|[API-Bank](https://arxiv.org/abs/2304.08244) (Li et al.)|2023|[Code](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)|
|[T-Eval](https://arxiv.org/abs/2312.14033) (Chen et al.)|2023|[Code](https://github.com/open-compass/T-Eval)|
|[RoTBench](https://arxiv.org/abs/2401.08326) (Ye et al.)|2024|[Code](https://github.com/JunjieYe/RotBench)|
|[AgentBoard](https://arxiv.org/abs/2401.13178) (Ma et al.)|2024|[Code](https://github.com/hkust-nlp/AgentBoard)|
|[ToolEyes](https://arxiv.org/abs/2401.00741) (Ye et al.)|2024|[Code](https://github.com/JunjieYe/ToolEyes)|
|[MetaTool](https://arxiv.org/abs/2310.03128) (Huang et al.)|2023|[Code](https://github.com/meta-toolbench/MetaTool)|

### Function Calling Specific
|Name|Paper|Year|Code|
|---|---|---|---|
|[BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)|2024|[Code](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)|
|[NESTFUL](https://arxiv.org/abs/2409.03797) (Basu et al.)|2024|[Code](https://github.com/IBM/Nestful)|
|[NesTools](https://arxiv.org/abs/2410.11805) (Han et al.)|2024|[Code](https://github.com/ZurichNLP/NesTools)|
|[ShortcutsBench](https://arxiv.org/abs/2407.00132) (Shen et al.)|2024|[Code](https://github.com/microsoft/ShortcutsBench)|
|[API-BLEND](https://arxiv.org/abs/2402.15491) (Basu et al.)|2024|[Dataset](https://huggingface.co/datasets/ibm-granite/API-BLEND)|
|[Seal-Tools](https://arxiv.org/abs/2405.08355) (Wu et al.)|2024|[Code](https://github.com/tonyzhaozh/seal-tools)|

## Applications

### Retrieval-Augmented Generation
|Paper|Venue|Year|Code|
|---|---|---|---|
|[A survey on RAG meets LLMs: Towards retrieval-augmented large language models](https://arxiv.org/abs/2405.06211) (Ding et al.)|arXiv|2024||
|[ChatDB: Augmenting LLMs with Databases as Their Symbolic Memory](https://arxiv.org/abs/2306.03901) (Hu et al.)|arXiv|2023|[Code](https://github.com/huchenxucs/ChatDB)|
|[LLM-dCache: Improving Tool-Augmented LLMs with GPT-Driven Localized Data Caching](https://arxiv.org/abs/2406.06799) (Singh et al.)|arXiv|2024||
|[AvaTaR: Optimizing LLM Agents for Tool-Assisted Knowledge Retrieval](https://arxiv.org/abs/2406.11200) (Wu et al.)|arXiv|2024|[Code](https://github.com/shirleyisw/AvaTaR)|

### Domain-Specific Applications
|Paper|Venue|Year|Code|
|---|---|---|---|
|[Tool Calling: Enhancing Medication Consultation via Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2404.17897) (Huang et al.)|arXiv|2024||
|[Alchemist: LLM-Aided End-User Development of Robot Applications](https://ieeexplore.ieee.org/document/10444584) (Karli et al.)|HRI|2024||
|[TOOL-ED: Enhancing Empathetic Response Generation with the Tool Calling Capability of LLM](https://arxiv.org/abs/2412.03096) (Cao et al.)|arXiv|2024||
|[Exploring Large Language Models in Financial Argument Relation Identification](https://aclanthology.org/2024.finnlp-1.12/) (Otiefy et al.)|FinNLP@LREC-COLING|2024||
|[AppAgent: Multimodal agents as smartphone users](https://arxiv.org/abs/2312.13771) (Yang et al.)|arXiv|2023|[Code](https://github.com/mnotgod/AppAgent)|
|[OS-ATLAS: A Foundation Action Model for Generalist GUI Agents](https://arxiv.org/abs/2410.23218) (Wu et al.)|arXiv|2024|[Code](https://github.com/microsoft/OS-ATLAS)|

### Multi-Agent Systems
|Paper|Venue|Year|Code|
|---|---|---|---|
|[MetaAgents: Simulating interactions of human behaviors for LLM-based task-oriented coordination](https://arxiv.org/abs/2310.06500) (Li et al.)|arXiv|2023||
|[AgentOhana: Design Unified Data and Training Pipeline for Effective Agent Learning](https://arxiv.org/abs/2402.15506) (Zhang et al.)|arXiv|2024|[Code](https://github.com/DeepWok/agentohana)|
|[Efficient Multi-Agent Collaboration with Tool Use for Online Planning](https://arxiv.org/abs/2412.20145) (Zhou et al.)|arXiv|2024||
|[TravelPlanner: A Benchmark for Real-World Planning with Language Agents](https://arxiv.org/abs/2402.01622) (Xie et al.)|ICML|2024|[Code](https://github.com/jiangjiechen/TravelPlanner)|






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

## Evaluation

This section covers methods for assessing function calling capabilities in LLMs and major benchmarks for standardized evaluation.

### Overall Performance Metrics

| Metric Type | Description | Examples |
|-------------|-------------|----------|
| **Pass Rate** | Measures whether function calls execute successfully | [ToolLLM](https://arxiv.org/abs/2307.16789), [NESTFUL](https://arxiv.org/abs/2409.03797) |
| **Win/Success Rate** | Evaluates quality of solutions beyond just execution | Information richness, Factual accuracy, API efficiency metrics |
| **Quality Metrics** | Text generation metrics for response quality | [BLEU](https://aclanthology.org/P02-1040.pdf), [ROUGE-L](https://aclanthology.org/W04-1013.pdf), Exact Match, [F1 score](https://arxiv.org/abs/2402.15491) |
| **Comprehensive Assessment** | Multi-dimensional evaluation of function calling | [T-Eval](https://arxiv.org/abs/2312.14033) |

### Benchmarks

| Category | Description | Examples |
|----------|-------------|----------|
| **Foundational Benchmarks** | Early standardized evaluation frameworks | [ToolLLM](https://arxiv.org/abs/2307.16789), [ToolAlpaca](https://arxiv.org/abs/2306.05301), [Gorilla](https://arxiv.org/abs/2305.15334) |
| **Standardized Platforms** | Comprehensive evaluation infrastructures | [API-Bank](https://arxiv.org/abs/2304.08244), [APIBench](https://arxiv.org/abs/2305.15334) |
| **Domain-Specific Benchmarks** | Evaluations for specialized application areas | [ShortcutsBench](https://arxiv.org/abs/2407.00132), [BigCodeBench](https://arxiv.org/abs/2406.15877), [SEAL](https://arxiv.org/abs/2409.15523), [RadABench](https://arxiv.org/abs/2412.09529) |
| **Task-Oriented Frameworks** | Focus on specific functional aspects | [IN3](https://arxiv.org/abs/2402.09205), [NESTFUL](https://arxiv.org/abs/2409.03797), [UltraTool](https://arxiv.org/abs/2404.00450), [TravelPlanner](https://openreview.net/forum?id=tmlBjf2zCo), [ChinaTravel](https://arxiv.org/abs/2412.13682) |
| **Comprehensive Systems** | Holistic evaluation across multiple dimensions | [API-BLEND](https://arxiv.org/abs/2402.15491), [NESTOOLS](https://arxiv.org/abs/2410.11805), [MTU-Bench](https://arxiv.org/abs/2410.11710), [WTU-EVAL](https://arxiv.org/abs/2407.12823) |

## Industry Products

This section outlines commercial platforms, development frameworks, autonomous agent systems, and open-source models implementing function calling capabilities.

### Commercial Platforms

| Platform | Description | Link |
|----------|-------------|------|
| **ChatGPT Plugins** | OpenAI's ecosystem for extending ChatGPT with external functions | [Documentation](https://openai.com/blog/chatgpt-plugins) |
| **Claude's Tool Use API** | Anthropic's function calling capabilities for Claude | [Documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) |
| **Cohere Command** | APIs like Command R and Command R+ for function integration | [Documentation](https://docs.cohere.com/v2/docs/command-r-plus) |
| **Qwen** | Alibaba Cloud's models with function calling support | [Documentation](https://qwen.readthedocs.io/en/latest/framework/function_call.html) |
| **DeepSeek** | Function calling API for DeepSeek models | [Documentation](https://api-docs.deepseek.com/guides/function_calling) |

### Development Frameworks and SDKs

| Framework | Description | Link |
|-----------|-------------|------|
| **HuggingFace Transformer Agents** | Framework for multimodal tasks using pre-trained models | [Documentation](https://huggingface.co/docs/transformers/transformers_agents) |
| **Semantic Kernel** | SDK for integrating LLMs with conventional programming | [GitHub](https://github.com/microsoft/semantic-kernel) |
| **LangChain** | Framework for creating LLM-powered applications | [GitHub](https://github.com/langchain-ai/langchain) |
| **WebCPM** | Framework for Chinese long-form QA with web search | [Paper](https://arxiv.org/abs/2305.06849) |

### Autonomous Agent Systems

| System | Description | Link |
|--------|-------------|------|
| **Auto-GPT** | Autonomous LLM-based agent system | [GitHub](https://github.com/Significant-Gravitas/AutoGPT) |
| **BabyAGI** | Framework for autonomous AI agents | [GitHub](https://github.com/yoheinakajima/babyagi) |
| **BMTools** | Repository for enhancing LLMs with various tools | [GitHub](https://github.com/OpenBMB/BMTools) |
| **RestGPT** | Framework for LLMs to interact with RESTful APIs | [Paper](https://arxiv.org/abs/2306.06624) |
| **xLAM** | Advanced approach for tool usage | [Dataset](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) |

### Open Source Models

| Model | Description | Link |
|-------|-------------|------|
| **GRANITE-20B-FUNCTIONCALLING** | IBM's function calling model | [Paper](https://arxiv.org/abs/2407.00121) |
| **Mistral 7B** | Open-source model with function calling capabilities | [Paper](https://arxiv.org/abs/2310.06825) |
| **NexusRaven V2-13B** | Open-source LLM for advanced function calling | [GitHub](https://huggingface.co/NexusRaven/NexusRaven-13B) |
| **Gorilla OpenFunctions** | Specialized in API interactions | [Paper](https://arxiv.org/abs/2305.15334) |
| **FireFunction V1** | Function calling optimized model | [HuggingFace](https://huggingface.co/fireworks-ai/firefunction-v1) |
| **Nous Hermes 2** | Enhanced performance in function calling | [HuggingFace](https://huggingface.co/NousResearch/Nous-Hermes-13b) |

## Open Issues and Future Directions

This section outlines critical challenges and future research directions for function calling in LLMs.

### Service Issues of Function Calling

| Issue | Description |
|-------|-------------|
| **Standardization Challenges** | Lack of universal standards for assessing function call quality across services |
| **Latency and Throughput** | High latency and low throughput, especially when integrating tool learning into reasoning |
| **Security Vulnerabilities** | Recent research identifying security risks like "jailbreak function" attacks with high success rates |
| **Quality Assessment** | Need for integrated frameworks evaluating response time, accuracy, user satisfaction, and security |

### Usability and Modification of Functions

| Issue | Description |
|-------|-------------|
| **Technical and Cost Barriers** | Challenges in identifying which functions to utilize and modify in specific scenarios |
| **Integration Complexity** | Difficulties ensuring data format consistency and overcoming system architecture limitations |
| **Standardization Needs** | Requirement for standardized API modification processes covering evaluation, design, testing, and deployment |
| **Cost-Benefit Balance** | Need for clear assessment models considering both direct and indirect costs of modifications |

### Feedback Quality and Optimization

| Issue | Description |
|-------|-------------|
| **Complex Feedback Processing** | Limited understanding of how LLMs learn from human feedback in complex function-calling scenarios |
| **Ambiguous Feedback Interpretation** | Challenges in accurately processing unstructured or ambiguous user feedback |
| **Strategy Development** | Need for advanced algorithms to interpret intent behind feedback and handle uncertainty |
| **Learning Assessment** | Difficulty in quantifying improvement based on varied types of human feedback |

### Function Isolation and Post-Processing

| Issue | Description |
|-------|-------------|
| **Isolation Requirements** | Need to appropriately isolate functions for business needs and regulatory compliance |
| **Implementation Complexity** | Challenges in flexible design and high customization for post-processing |
| **Compliance Management** | Balancing innovation with adherence to regulations across different scenarios |
| **Efficiency Maintenance** | Ensuring isolation and post-processing don't negatively impact service efficiency |

## Contributing

Contributions to this repository are welcome! Please submit a pull request or open an issue to suggest additions or improvements.

## License

This repository is licensed under [MIT License](LICENSE).
