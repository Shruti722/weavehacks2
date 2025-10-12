# SynErgi - Multi-Agent Grid Optimization System

A multi-agent reinforcement learning system for optimizing power grid operations in San Francisco, using LangGraph for agent orchestration and W&B Inference for LLM-powered reasoning.

## Overview

SynErgi uses three specialized AI agents that collaborate to analyze, plan, and execute grid optimization decisions:

- **Analyst Agent**: Analyzes grid state, identifies supply-demand imbalances, and detects critical issues
- **Planner Agent**: Creates action plans to address grid issues while prioritizing equity-weighted zones
- **Actuator Agent**: Translates plans into specific execution instructions with equipment, timing, and safety checks

## Features

- 🤖 **Multi-Agent Workflow**: LangGraph-based agent collaboration with Digital Twin → Analyst → Planner → Actuator flow
- 🧠 **LLM-Powered Reasoning**: Uses Qwen2.5 14B Instruct via W&B Inference (optimized for future RL fine-tuning)
- ⚡ **Real-Time Grid Simulation**: 45 San Francisco neighborhoods with realistic demand/supply patterns
- 📊 **Comprehensive Analytics**: Risk assessment, fairness metrics, and equity-weighted prioritization
- 🔄 **Full Conversation Tracking**: Monitor agent-to-agent communication and reasoning

## Architecture

```
┌─────────────────┐
│  Digital Twin   │ (Grid State Data)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyst Agent   │ (Analyze Issues)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Planner Agent   │ (Create Action Plan)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Actuator Agent  │ (Execute Plan)
└─────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- W&B account with API key ([get here](https://wandb.ai/authorize))

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/imruchi/weavehacks2.git
   cd weavehacks2
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your WANDB_API_KEY
   ```

## Usage

### Run the Multi-Agent System

**Default test:**
```bash
python agents/testing.py
```

**Custom prompt:**
```bash
python agents/testing.py "Optimize battery usage in Tenderloin and Bayview"
```

**Example prompts:**
- `"Find high-risk nodes and reduce overload"`
- `"Prioritize equity zones while minimizing costs"`
- `"Balance supply-demand across all districts"`

### Expected Output

The system will:
1. Generate realistic grid state (1500-1600 MW demand across 45 nodes)
2. Run agents through the workflow
3. Display detailed analysis, plan, and execution steps
4. Show full agent conversation transcript

Example output:
```
======================================================================
SynErgi Multi-Agent System - Test
======================================================================

📝 User Prompt: Analyze the grid and create a plan...

🔄 Generating grid state data...
   Time: 19:57
   Demand: 1593.0 MW
   Supply: 1524.9 MW
   Risk: 0.105

🤖 Initializing Multi-Agent System...
   ✓ System initialized

⚙️  Running Agent Workflow...
   Digital Twin → Analyst → Planner → Actuator

======================================================================
📊 RESULTS
======================================================================

🔍 ANALYST ANALYSIS:
The current grid state indicates a significant city-wide supply-demand
imbalance, with a total demand of 1593.0 MW exceeding supply by 68.1 MW...

📋 PLANNER ACTION PLAN:
Based on the analysis, here is the action plan:
- Financial District: +10 MW supply increase (gas turbine)
- Tenderloin: +2 MW discharge storage (battery)
- Russian Hill: -2 MW demand reduction (demand response)...

⚡ ACTUATOR EXECUTION:
Execution instructions:
- Start gas turbine in Financial District (0.5 MW/min ramp rate)
- Discharge battery in Tenderloin (5 min operation)
- Activate demand response in Russian Hill...
```

## Configuration

### Model Options

Edit `multi_agent_system.py` to change the LLM model:

```python
# Default: Qwen2.5 14B (recommended for RL)
mas = SynErgiMultiAgentSystem(model_name="Qwen/Qwen2.5-14B-Instruct")

# Alternatives:
# - meta-llama/Llama-3.1-8B-Instruct (lighter, 8B params)
# - microsoft/Phi-4-mini-instruct (very light, 3.8B params)
```

### Environment Variables

Create a `.env` file with:
```bash
WANDB_API_KEY=your_api_key_here
WEAVE_PROJECT=synergi-grid-optimization
```

## Project Structure

```
weavehacks2/
├── agents/
│   ├── analyst_agent.py      # Grid analysis agent
│   ├── planner_agent.py      # Action planning agent
│   ├── actuator_agent.py     # Execution agent
│   ├── testing.py            # Test script
│   └── tools/                # Agent tools (future use)
├── data/
│   ├── simulator.py          # Grid state simulator
│   ├── data.json             # SF grid template
│   └── current_state.json    # Latest simulation
├── multi_agent_system.py     # LangGraph orchestration
├── requirements.txt          # Dependencies
└── README.md
```

## Technical Details

### Grid Simulator

Generates realistic power grid data for 45 SF neighborhoods:
- Dynamic demand/supply following time-of-day patterns
- Weather simulation (temperature, solar, wind)
- Real-time pricing (day-ahead and real-time)
- Storage state of charge (SOC)
- Risk indicators (overload, N-1 margin)
- Fairness metrics across equity zones

### Agent Communication

Agents communicate through LangGraph state:
- **Digital Twin** provides initial grid state
- **Analyst** receives state, outputs analysis
- **Planner** receives analysis, outputs action plan
- **Actuator** receives plan, outputs execution instructions

Each agent sees the full conversation history.

### Model Selection

**Qwen2.5 14B Instruct** is used because:
- ✅ 14B parameters (manageable for RL fine-tuning with LoRA/QLoRA)
- ✅ Strong reasoning capabilities
- ✅ Supports tool use and multi-turn conversations
- ✅ Free tier on W&B Inference
- ✅ Good balance of performance vs. trainability

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and create a Pull Request

## License

MIT License

## Acknowledgments

- Built for Weave Hacks 2025
- Uses [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- Powered by [W&B Inference](https://wandb.ai/site/inference) for LLM hosting
- Grid data inspired by San Francisco's power infrastructure

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Status**: 🚧 Active Development | **Model**: Qwen2.5 14B | **Last Updated**: October 2025
