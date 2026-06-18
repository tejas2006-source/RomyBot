# RomyBot

RomyBot is a trading agent project I’m building to explore algorithmic trading, quantitative research, and AI-driven decision systems. The goal is to create a modular system that can ingest market data, test strategies, evaluate performance, and eventually support more intelligent decision-making components.

## Why I’m building this

I’m a Computer Science and Engineering student at UC San Diego interested in AI, agents, and high-performance technical systems. RomyBot is one of my core portfolio projects for learning:
- Python engineering
- quantitative analysis
- backtesting and evaluation
- machine learning for finance
- building real-world developer workflows on GitHub

## Current status

This project is in active development. The first milestone is a clean backtesting system with simple rule-based strategies before expanding into ML-based models and agent behavior.

## Planned features

- Historical market data ingestion
- Strategy framework for testing trading ideas
- Backtesting engine
- Performance metrics dashboard
- Risk management rules
- Paper trading simulation
- Optional machine learning forecasting module
- Long-term: agentic decision layer for strategy selection and monitoring

## Tech stack

- Python
- Pandas
- NumPy
- Matplotlib / Plotly
- scikit-learn
- APIs for market data
- Git + GitHub for version control

## Roadmap

### Phase 1: Foundations
- Set up repository structure
- Add README and license
- Create virtual environment and requirements
- Load and clean historical price data
- Implement first simple strategies

### Phase 2: Backtesting
- Build trade execution logic
- Add portfolio tracking
- Compute return, drawdown, Sharpe ratio, win rate
- Compare strategies on historical datasets

### Phase 3: Intelligence
- Add predictive features
- Experiment with ML models
- Evaluate whether ML improves performance
- Document results clearly

### Phase 4: Agent layer
- Add higher-level orchestration logic
- Enable monitoring, strategy switching, and reporting
- Explore whether LLM/agent workflows can help with research or analysis

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/romybot.git
cd romybot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
