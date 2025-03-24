```markdown
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
```

Now I'll continue with the Challenges section, following the structure from your taxonomy:

```markdown
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
```

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
| Token Representation | [ToolGen: Unified Tool Retrieval and Calling via Generation](https://arxiv.org/abs/2410.03439) (Wang et al., 2024) | [Code](https://github.com/OpenLLMAI/ToolGen) | Encodes functions as special tokens during training for efficiency, converted to natural language during inference |
| Multi-turn Interaction | [Sequential API Function Calling Using GraphQL Schema](https://aclanthology.org/2024.emnlp-industry-4107/) (Saha et al., 2024) | - | Introduces structured API schemas and response mapping for sequential function calling |
| Function Masking | [Hammer: Robust Function-Calling for On-Device Language Models via Function Masking](https://arxiv.org/abs/2410.04587) (Lin et al., 2024) | - | Specialized techniques to address naming convention sensitivity issues for on-device deployment |

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
