# Comprehensive Reviews into Function Calling in Large Language Models

An index of concepts, frameworks, and methodologies in:
- **Function Calling Pipeline**: Understanding the entire process from pre-call to post-call stages
- **Sample Construction & Fine-tuning**: Building effective training datasets and optimizing models
- **Deployment & Inference**: Practical implementation strategies for real-world applications
- **Evaluation Frameworks**: Benchmarks and metrics for assessing function calling capabilities

Reproducibility is important! We prioritize methods with open-source implementations.

Please cite our survey paper if this index is helpful:

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
- [Challenges](#challenges)
- [Sample Construction and Fine-Tuning](#sample-construction-and-fine-tuning)
- [Deployment and Inference](#deployment-and-inference)
- [Evaluation](#evaluation)
- [Industry Products](#industry-products)
- [Open Issues](#open-issues)


## Challenges

### Pre-call Stage
| Challenge | Description |
|-----------|-------------|
| Challenge 1.1: Intent Recognition | Understanding user intentions accurately from natural language queries |
| Challenge 1.2: Function Redundancy | Managing redundant functions that serve similar purposes, increasing selection complexity |

### On-call Stage
| Challenge | Description |
|-----------|-------------|
| Challenge 2.1: Missing Calls | Failure to initiate function calls when required for task completion |
| Challenge 2.2: Unnecessary Calls | Triggering function calls when not required by the user's task |
| Challenge 3.1: Missing/Illegal Parameters | Inadequate or inappropriate parameter extraction from user inputs |
| Challenge 3.2: Function Hallucination | Mistakenly calling non-candidate or non-existent functions |
| Challenge 3.3: Pronouns Resolving | Correctly interpreting contextual references and pronouns in queries |
| Challenge 3.4: LLM Inherent Limitations | Performance constraints in latency and accuracy due to model architecture |
| Challenge 3.5: Multi-Call Procedure | Managing complex workflows requiring multiple related function calls |
| Challenge 3.6: Effective Context Management | Maintaining relevant information across multi-turn conversations |

### Post-call Stage
| Challenge | Description |
|-----------|-------------|
| Challenge 4.1: Execution Result Mismatch | Function outputs not aligning with user expectations |
| Challenge 4.2: Irrelevant Information Overload | Excessive irrelevant information in function outputs |
| Challenge 4.3: Mismatch Between Real-World Functions and Results | Gap between LLM-generated outputs and executable code |
| Challenge 4.4: Execution Failure | Functions failing despite correct triggering and parameterization |


## Sample Construction and Fine-Tuning

### Function Collection
| Method | Description |
|--------|-------------|
| Manual Construction | Human-crafted functions with precise specifications and documentation |
| LLM Generation | Leveraging large language models like GPT-4, LlaMA 70B, and Qwen to automatically generate function specifications |
| Web Mining | Extracting diverse function objects from web resources, with descriptions supplemented by LLMs when necessary |

### Sample Construction
| Approach | Paper | Code | Description |
|----------|-------|------|-------------|
| Text Representation | [Toolformer: Language models can teach themselves to use tools](https://arxiv.org/abs/2302.04761) (Schick et al., 2024) | [Code](https://github.com/lucidrains/toolformer-pytorch) | Represents functions as natural language text, providing flexibility but requiring more token space |
| Text Representation | [ToolGen: Unified Tool Retrieval and Calling via Generation](https://arxiv.org/abs/2410.03439) (Wang et al., 2024) | [Code](https://github.com/OpenLLMAI/ToolGen) | Integrates tool information through generation with natural language descriptions |
| Token Representation | [Toolformer: Language models can teach themselves to use tools](https://arxiv.org/abs/2302.04761) (Schick et al., 2024) | [Code](https://github.com/lucidrains/toolformer-pytorch) | Encodes functions as special tokens during training for computational efficiency |
| Token Representation | [ToolGen: Unified Tool Retrieval and Calling via Generation](https://arxiv.org/abs/2410.03439) (Wang et al., 2024) | [Code](https://github.com/OpenLLMAI/ToolGen) | Uses token representation during training while maintaining semantic richness |
| Multi-turn Interaction | [Sequential API Function Calling Using GraphQL Schema](https://aclanthology.org/2024.emnlp-industry-4107/) (Saha et al., 2024) | - | Introduces structured API schemas and response mapping for sequential function calling |
| Multi-turn Interaction | [Hammer: Robust Function-Calling for On-Device Language Models via Function Masking](https://arxiv.org/abs/2410.04587) (Lin et al., 2024) | - | Specialized techniques to address naming convention sensitivity issues for on-device deployment |

### Fine-tuning Strategies
| Method | Papers | Description |
|--------|--------|-------------|
| Supervised Fine-Tuning (SFT) | [Show your work: Scratchpads for intermediate computation with language models](https://arxiv.org/abs/2112.00114) (Nye et al., 2021), [Giving BERT a calculator: Finding operations and arguments with reading comprehension](https://arxiv.org/abs/1909.00109) (Andor et al., 2019), [Rainier: Reinforced knowledge introspector for commonsense question answering](https://arxiv.org/abs/2210.03078) (Liu et al., 2022) | Standard fine-tuning approach that maximizes likelihood of correct function calls and parameters |
| Parameter-Efficient Fine-Tuning (PEFT) | [Gpt4tools: Teaching large language model to use tools via self-instruction](https://arxiv.org/abs/2305.05181) (Yang et al., 2024), [CITI: Enhancing Tool Utilizing Ability in Large Language Models without Sacrificing General Performance](https://arxiv.org/abs/2409.13202) (Hao et al., 2024) | Techniques like LoRA that update only a small subset of model parameters |
| Reinforcement Learning & RLHF | [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) (Nakano et al., 2021), [Taskmatrix.ai: Completing tasks by connecting foundation models with millions of apis](https://arxiv.org/abs/2303.16434) (Liang et al., 2024) | Learning from user feedback and environment interactions to improve function calling decisions |

### Critical Emphasis
| Emphasis | Description |
|----------|-------------|
| Data Quality | Prioritizing dataset diversity and verification over quantity for more robust function calling capabilities |
| Model Scaling | Larger models demonstrate significantly better function calling capabilities, with notable improvements above 7B parameters |
| Capability Balance | Maintaining a balance between specialized function calling abilities and general language capabilities to avoid performance tradeoffs |

## Deployment and Inference

### Task Planning

#### Foundational Planning Mechanisms
| Name | Paper | Venue | Code | Comment |
|------|-------|-------|------|---------|
| ReAct | [React: Synergizing reasoning and acting in language models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022) | NeurIPS | [Code](https://github.com/ysymyth/ReAct) | Combines reasoning and acting through chain-of-thought prompts |
| ToolFormer | [Toolformer: Language models can teach themselves to use tools](https://arxiv.org/abs/2302.04761) (Schick et al., 2023) | NeurIPS | [Code](https://github.com/lucidrains/toolformer-pytorch) | Enables LLMs to use external tools through self-supervised learning |
| Reverse Chain | [Reverse chain: A generic-rule for llms to master multi-api planning](https://arxiv.org/abs/2310.04474) (Zhang et al., 2023) | arXiv | - | Introduces target-driven backward reasoning for controlled multi-API planning |
| Agent Laboratory | [Agent laboratory: Using llm agents as research assistants](https://arxiv.org/abs/2501.04227) (Schmidgall et al., 2025) | arXiv | - | Multi-agent architecture with specialized roles for research planning |
| AVATAR | [AvaTaR: Optimizing LLM Agents for Tool-Assisted Knowledge Retrieval](https://arxiv.org/abs/2406.11200) (Wu et al., 2024) | arXiv | [Code](https://github.com/Shirley-Kokane/AvaTaR) | Actor-comparator architecture for tool-assisted knowledge retrieval |
| DEPS | [Describe, explain, plan and select: interactive planning with LLMs enables open-world multi-task agents](https://arxiv.org/abs/2402.07152) (Wang et al., 2024) | NeurIPS | [Code](https://github.com/microsoft/DEPS) | Interactive planning through description-based decomposition |

#### GUI-based Approaches
| Name | Paper | Venue | Code | Comment |
|------|-------|-------|------|---------|
| AppAgent | [Appagent: Multimodal agents as smartphone users](https://arxiv.org/abs/2312.13771) (Yang et al., 2023) | arXiv | [Code](https://github.com/mnotgod96/AppAgent) | Two-stage training for mobile application interaction |
| OS-ATLAS | [OS-ATLAS: A Foundation Action Model for Generalist GUI Agents](https://arxiv.org/abs/2410.23218) (Wu et al., 2024) | arXiv | [Code](https://github.com/OSAgent/OS-ATLAS) | Unified action space across different GUI platforms |
| Ponder & Press | [Ponder & Press: Advancing Visual GUI Agent towards General Computer Control](https://arxiv.org/abs/2412.01268) (Wang et al., 2024) | arXiv | [Code](https://github.com/yiqin001/Ponder-Press) | Two-stage planning with visual input for general computer control |

#### System Optimizations
| Name | Paper | Venue | Code | Comment |
|------|-------|-------|------|---------|
| Orca | [Orca: Progressive learning from complex explanation traces of gpt-4](https://arxiv.org/abs/2306.02707) (Mukherjee et al., 2023) | arXiv | - | Specialized approach for optimizing LLM inference |
| Memgpt | [Memgpt: Towards llms as operating systems](https://arxiv.org/abs/2310.08560) (Packer et al., 2023) | arXiv | [Code](https://github.com/cpacker/MemGPT) | Operating system design for LLM workloads |
| LLM-Tool Compiler | [An LLM compiler for parallel function calling](https://arxiv.org/abs/2312.04511) (Kim et al., 2023) | arXiv | [Code](https://github.com/SqueezeAILab/LLMCompiler) | Fuses similar tool operations for parallel execution |

#### Error Handling Approaches
| Name | Paper | Venue | Code | Comment |
|------|-------|-------|------|---------|
| LLM-Planner | [Llm-planner: Few-shot grounded planning for embodied agents with large language models](https://arxiv.org/abs/2212.04088) (Song et al., 2023) | ICCV | [Code](https://github.com/OSU-NLP-Group/LLM-Planner) | Environmental feedback for plan regeneration during failures |
| ToolChain* | [Toolchain*: Efficient action space navigation in large language models with a* search](https://arxiv.org/abs/2310.13227) (Zhuang et al., 2023) | arXiv | [Code](https://github.com/yzhuang0222/ToolChain) | Employs decision trees for systematic API call management |
| AMOR | [AMOR: A Recipe for Building Adaptable Modular Knowledge Agents Through Process Feedback](https://arxiv.org/abs/2402.01469) (Guan et al., 2024) | arXiv | [Code](https://github.com/gpt4life/amor) | FSM-based framework enabling process-level human feedback |

#### Tree-based Approaches
| Name | Paper | Venue | Code | Comment |
|------|-------|-------|------|---------|
| ControlLLM | [Controlllm: Augment language models with tools by searching on graphs](https://arxiv.org/abs/2310.17796) (Liu et al., 2023) | arXiv | [Code](https://github.com/OpenGVLab/ControlLLM) | Tree of Thoughts with depth-first search on tool graphs |
| Toolink | [Toolink: Linking toolkit creation and using through chain-of-solving on open-source model](https://arxiv.org/abs/2310.05155) (Qian et al., 2023) | arXiv | [Code](https://github.com/toolllm/toolink) | Hierarchical task decomposition with toolkit creation |
| α-UMi | [Small llms are weak tool learners: A multi-llm agent](https://arxiv.org/abs/2401.07324) (Shen et al., 2024) | arXiv | [Code](https://github.com/zjunlp/AntGPT) | Planning-oriented fine-tuning for small LLMs |

#### Adaptive Planning Strategies
| Name | Paper | Venue | Code | Comment |
|------|-------|-------|------|---------|
| Agent-Pro | [Agent-pro: Learning to evolve via policy-level reflection and optimization](https://arxiv.org/abs/2402.17574) (Zhang et al., 2024) | arXiv | [Code](https://github.com/sail-sg/Agent-Pro) | Dynamic belief management and policy-level reflection |
| Inner Thoughts | [Proactive Conversational Agents with Inner Thoughts](https://arxiv.org/abs/2501.00383) (Liu et al., 2024) | arXiv | [Code](https://github.com/hkust-nlp/PCA) | Continuous thought generation for proactive participation |
| PAE | [Proposer-Agent-Evaluator (PAE): Autonomous Skill Discovery For Foundation Model Internet Agents](https://arxiv.org/abs/2412.13194) (Zhou et al., 2024) | arXiv | [Code](https://github.com/LifeLongTeam/pae-autonomous-skill-discovery) | Context-aware task proposal and evaluation |

### Prompt Construction
| Approach | Paper | Code | Comment |
|----------|-------|------|---------|
| Few-shot Integration | [Nexusraven: a commercially-permissive language model for function calling](https://arxiv.org/abs/2311.11981) (Srinivasan et al., 2023) | [Code](https://github.com/nexusflow/NexusRaven) | Utilizes 16 examples per API function with four-shot prompting |
| Context Management | - | - | Includes function definitions, docstrings, and chain-of-thought explanations |
| Query-based Retrieval | [Learning to Ask: When LLMs Meet Unclear Instruction](https://arxiv.org/abs/2409.00557) (Wang et al., 2024) | [Code](https://github.com/CAISA-Lab/UnclearInstrLearningToAsk) | Encourages proactive question-asking before API calls |

### Function Generation
| Approach | Paper | Venue | Code | Comment |
|----------|-------|-------|------|---------|
| Grammar Control | [Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047) (Park et al., 2024) | arXiv | [Code](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic_kernel/core_plugins) | Constrains output using context-free grammar |
| TOOL-ED | [TOOL-ED: Enhancing Empathetic Response Generation with the Tool Calling Capability of LLM](https://arxiv.org/abs/2412.03096) (Cao et al., 2024) | arXiv | - | Treats knowledge bases as callable tools for empathetic dialogue |
| IBSEN | [IBSEN: Director-Actor Agent Collaboration for Controllable and Interactive Drama Script Generation](https://aclanthology.org/2024.acl-long.99/) (Han et al., 2024) | ACL | [Code](https://github.com/X-PLUG/IBSEN) | Multi-agent coordination for controlled script generation |

### Function Mapping
| Approach | Paper | Code | Comment |
|----------|-------|------|---------|
| Rule-based Resolution | [Deterministic coreference resolution based on entity-centric, precision-ranked rules](https://aclanthology.org/J13-4004/) (Lee et al., 2013) | [Code](https://github.com/huggingface/neuralcoref) | Predefined mapping rules for contextual references |
| Knowledge Reasoning | [Knowledge-aware Pronoun Coreference Resolution](https://aclanthology.org/P19-1081/) (Zhang et al., 2019) | - | Leverages knowledge graphs for reference resolution |
| LLM Mapping | [End-to-end Neural Coreference Resolution](https://aclanthology.org/D17-1018/) (Lee et al., 2017) | [Code](https://github.com/kentonl/e2e-coref) | Uses neural models for contextual mapping |
| Dictionary Mapping | [Syllabus: Portable Curricula for Reinforcement Learning Agents](https://arxiv.org/abs/2411.11318) (Sullivan et al., 2024) | [Code](https://github.com/rsullivan00/syllabus) | Unified APIs and format alignment mechanisms |
| Parameter Validation | - | - | Verification of parameter completeness and accuracy |

### Response Generation
| Approach | Paper | Code | Comment |
|----------|-------|------|---------|
| Placeholder Results | [Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings](https://arxiv.org/abs/2305.11554) (Hao et al., 2024) | [Code](https://github.com/ryoungj/ToolkenGPT) | Generated placeholders replaced with API call results |
| Template Structuring | [Gorilla: Large language model connected with massive apis](https://arxiv.org/abs/2305.15334) (Patil et al., 2023) | [Code](https://github.com/ShishirPatil/gorilla) | Structured templates for consistent output formatting |
| Agent Correction | [Learning to use tools via cooperative and interactive agents](https://arxiv.org/abs/2403.03031) (Shi et al., 2024) | [Code](https://github.com/RUC-GSAI/LLM-CU) | Specialized agents review and correct each other's actions |
| RAG Integration | [Sfr-rag: Towards contextually faithful llms](https://arxiv.org/abs/2409.09916) (Nguyen et al., 2024) | [Code](https://github.com/rain-fish/sfr-rag) | Contextually faithful retrieval-augmented generation |

### Memory Scheme
| Approach | Paper | Code | Comment |
|----------|-------|------|---------|
| Hierarchical Memory | [Memorybank: Enhancing large language models with long-term memory](https://ojs.aaai.org/index.php/AAAI/article/view/29545) (Zhong et al., 2024) | [Code](https://github.com/zhongwanjun/MemoryBank-SiliconFriend) | Hierarchical storage with Ebbinghaus-inspired updating |
| Self-Controlled Memory | [Unleashing infinite-length input capacity for large-scale language models with self-controlled memory system](https://arxiv.org/abs/2304.10203) (Liang et al., 2023) | - | Memory management through control systems |
| Thought-based Storage | [Think-in-memory: Recalling and post-thinking enable llms with long-term memory](https://arxiv.org/abs/2311.08719) (Liu et al., 2023) | [Code](https://github.com/microsoft/ThinkInMemory) | Stores and recalls thoughts rather than raw conversations |
| Trajectory Framework | [Synapse: Trajectory-as-exemplar prompting with memory for computer control](https://arxiv.org/abs/2310.15142) (Zheng et al., 2023) | [Code](https://github.com/microsoft/Synapse) | Complete trajectories as exemplars for planning |

## Evaluation

### Overall Performance Metrics
| Metric | Description | Example Works |
|--------|-------------|--------------|
| Pass Rate | Proportion of successfully completed instructions | [Toolllm: Facilitating large language models to master 16000+ real-world apis](https://arxiv.org/abs/2307.16789) (Qin et al., 2023) |
| Win/Success Rate | Quality evaluation including information richness, factual accuracy | [NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls](https://arxiv.org/abs/2409.03797) (Basu et al., 2024) |
| Function Selection Metrics | Recall@K, NDCG@K, COMP@K for evaluating tool selection | [COLT: Towards Completeness-Oriented Tool Retrieval for Large Language Models](https://arxiv.org/abs/2405.16089) (Qu et al., 2024) |
| Quality-based Metrics | BLEU, ROUGE-L, Exact Match, F1 score | [T-eval: Evaluating the tool utilization capability of large language models step by step](https://aclanthology.org/2024.acl-long.591/) (Chen et al., 2024) |

### Benchmarks
| Name | Paper | Code | Description |
|------|-------|------|-------------|
| ToolLLM | [Toolllm: Facilitating large language models to master 16000+ real-world apis](https://arxiv.org/abs/2307.16789) (Qin et al., 2023) | [Code](https://github.com/OpenBMB/ToolBench) | Comprehensive benchmark for API utility |
| Gorilla | [Gorilla: Large language model connected with massive apis](https://arxiv.org/abs/2305.15334) (Patil et al., 2023) | [Code](https://github.com/ShishirPatil/gorilla) | Berkeley Function Calling Leaderboard |
| API-Bank | [Api-bank: A benchmark for tool-augmented llms](https://arxiv.org/abs/2304.08244) (Li et al., 2023) | [Code](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) | Comprehensive API interaction testing |
| ShortcutsBench | [Shortcutsbench: A large-scale real-world benchmark for api-based agents](https://arxiv.org/abs/2407.00132) (Shen et al., 2024) | [Code](https://github.com/MobileLLM/ShortcutsBench) | Real APIs from Apple's operating systems |
| API-BLEND | [API-BLEND: A Comprehensive Corpora for Training and Benchmarking API LLMs](https://arxiv.org/abs/2402.15491) (Basu et al., 2024) | [Code](https://github.com/IBM/API-Blend) | Multi-domain API coverage with evaluation methods |
| MTU-Bench | [MTU-Bench: A Multi-granularity Tool-Use Benchmark for Large Language Models](https://arxiv.org/abs/2410.11710) (Wang et al., 2024) | [Code](https://github.com/X-LANCE/MTU-Bench) | Multi-granularity tool-use evaluation |

## Industry Products

### Commercial Platforms
| Name | Documentation | Description |
|------|---------------|-------------|
| ChatGPT plugins | [Documentation](https://platform.openai.com/docs/plugins/introduction) | Extends ChatGPT with external application access |
| Claude's tool use | [Documentation](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) | Controlled interactions with external functions |
| Cohere Command | [Documentation](https://docs.cohere.com/docs/command-r-plus) | Command R+ with function calling capabilities |
| Qwen | [Documentation](https://qwen.readthedocs.io/en/latest/framework/function_call.html) | Open-source models with function calling support |

### Development Frameworks and SDKs
| Name | Documentation | Code | Description |
|------|---------------|------|-------------|
| HuggingFace Transformer Agents | [Documentation](https://huggingface.co/docs/transformers/transformers_agents) | [Code](https://github.com/huggingface/transformers) | Multimodal task integration for LLMs |
| Semantic Kernel | [Documentation](https://learn.microsoft.com/en-us/semantic-kernel/overview/) | [Code](https://github.com/microsoft/semantic-kernel) | Integrates LLMs with conventional programming languages |
| LangChain | [Documentation](https://python.langchain.com/docs/get_started/introduction) | [Code](https://github.com/langchain-ai/langchain) | Framework for LLM-powered applications |

### Autonomous Agent Systems
| Name | Documentation | Code | Description |
|------|---------------|------|-------------|
| Auto-GPT | [Documentation](https://docs.agpt.co/) | [Code](https://github.com/Significant-Gravitas/AutoGPT) | Autonomous LLM for complex tasks with minimal input |
| BabyAGI | [Documentation](https://github.com/yoheinakajima/babyagi#readme) | [Code](https://github.com/yoheinakajima/babyagi) | Task management with minimal oversight |
| BMTools | [Documentation](https://github.com/OpenBMB/BMTools#readme) | [Code](https://github.com/OpenBMB/BMTools) | Community-driven tool integration platform |

### Open Source Models
| Name | Documentation | Code | Description |
|------|---------------|------|-------------|
| GRANITE-20B-FUNCTIONCALLING | [Model Card](https://huggingface.co/IBM/granite-20b-function-calling) | [Code](https://github.com/IBM/granite) | State-of-the-art open-source function calling model |
| NexusRaven V2-13B | [Model Card](https://huggingface.co/Nexusflow/NexusRaven-V2-13B) | [Code](https://github.com/nexusflow/NexusRaven) | Specialized in cybersecurity tool and API invocation |
| FireFunction V1 | [Model Card](https://huggingface.co/fireworks-ai/firefunction-v1) | - | Built on Mistral 7B with structured information generation |

## Open Issues

### Service Issues of Function Calling
- **Standards Challenge**: Lack of universally accepted standard for assessing quality and performance
- **Latency Problems**: High latency and low throughput affecting user experience
- **Security Vulnerabilities**: Potential for "jailbreak function" attacks and other security concerns

### Usability and Modification of Functions
- **Technical Costs**: Integration and maintenance costs for API modifications
- **System Architecture Limitations**: Constraints imposed by existing system architectures
- **Standardization Needs**: Requirement for standardized API modification processes

### Feedback Quality and Optimization
- **Complex Processing**: Multiple steps in feedback processing introducing errors
- **Learning Assessment**: Difficulty in quantifying effectiveness of human feedback
- **Strategy Requirements**: Need for advanced algorithms to interpret unstructured feedback

### Function Isolation and Post-Processing
- **Isolation Strategy**: Challenges in appropriately isolating functions for business needs
- **Regulatory Compliance**: Meeting specific regulatory requirements across functions
- **Post-processing Solutions**: Implementing effective middleware for compliance and data transformation

