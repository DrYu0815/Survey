# awesome-llm-function-calling [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

An index of research and tools in
- LLM function calling: enabling language models to invoke external functions or APIs
- Tool use and learning: how LLMs understand and utilize various tools
- Agent frameworks: systems that orchestrate LLMs with tools

**Reproducibility is important!** We prioritize methods with open-source implementations unless they are significant survey/review papers.

If this index is helpful for your research, please consider citing our paper:

@article{your_citation_here,
title={Your Paper Title},
author={Your Name and Coauthors},
journal={Journal Name},
year={2025}
}

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
