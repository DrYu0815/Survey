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
