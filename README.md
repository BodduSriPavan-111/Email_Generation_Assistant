# Email Generation Assistant

An LLM-powered Email Generation Assistant that generates emails across scenarios and evaluates them using dual-judge scoring.

## Benchmark
![plot](https://drive.google.com/uc?id=13BqGK0ct9JGME33TXqG-w5OgqZQNSwmW)

## Key Features
- Dual-LLM generation (Mistral vs GPT)
- Few-shot + role-based prompt design
- LLM-as-a-judge evaluation (3 custom metrics)
- Dual-judge scoring to reduce bias
- Scenario-based benchmarking (10 real-world cases)

## Setup

### Installation

```py
pip install mistralai==1.9.11 groq==1.1.2 matplotlib==3.10.3
```

### Usage

1. Generate emails (Mistral + Qwen)
```py
python src/generate.py
```

2. Evaluate using 3 metrics & dual judges
```py
python src/evaluate.py    
```

3. Aggregate results & visualize
```py
python src/plot.py
```

## Project Structure
```
project/
├── src/
│   ├── generate.py        # Email generation pipeline (both models)
│   ├── evaluate.py        # Evaluation pipeline (3 metrics, dual judges)
│   └── plot.py            # Result aggregation and visualization
├── outputs/
│   ├── generated_outputs.json        # Raw generated emails (all scenarios)
│   ├── mistral_comparison.csv        # Mistral per-scenario scores
│   ├── mistral_final_summary.json    # Mistral aggregate scores
│   ├── qwen_comparison.csv           # Qwen per-scenario scores
│   ├── qwen_final_summary.json       # Qwen aggregate scores
│   └── final_comparison.png          # Final comparison chart
└── README.md
```
