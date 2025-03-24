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
