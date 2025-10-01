# Systematic Options Auto‑Hedging Engine

## 📌 Overview
This project implements a **systematic options auto‑hedging framework** designed to dynamically manage risk exposures in real time. It combines **options theory, stochastic modeling, execution simulation, and reproducible research infrastructure** to demonstrate how a modern quant desk might approach automated hedging of equity derivatives.

The engine is modular, auditable, and research‑grade — built to highlight both **theoretical mastery** and **production‑oriented engineering practices**.

---

## 🎯 Objectives
- **Dynamic Hedging:** Maintain delta‑neutral or gamma‑scaled exposures under evolving market conditions.  
- **Execution Simulation:** Model realistic fills, slippage, and transaction costs using a simulated limit order book.  
- **Performance Attribution:** Quantify hedging error, PnL decomposition, and cost efficiency across volatility regimes.  
- **Reproducibility:** Ensure every experiment is fully documented, versioned, and auditable.  

---

## 🧩 Core Components
### 1. Greeks Engine
- Computes Δ, Γ, Θ, Vega under **Black‑Scholes** and **Heston stochastic volatility** models.  
- Validated against closed‑form benchmarks and stress‑tested across volatility surfaces.  

### 2. Hedging Policies
- **Delta‑Neutral:** Classic hedge to eliminate first‑order exposure.  
- **Gamma‑Scaled:** Adjusts hedge aggressiveness based on convexity risk.  
- Configurable via YAML for experiment reproducibility.  

### 3. Execution Layer
- **Simulated Limit Order Book (LOB):** Models queue priority, spread dynamics, and liquidity depth.  
- **Cost Models:** Captures slippage, bid‑ask spread, and market impact.  

### 4. Metrics & Attribution
- **Hedging Error:** Tracks deviation from target exposures.  
- **PnL Attribution:** Separates gains/losses into hedge efficiency, transaction costs, and residual risk.  
- **Scenario Analysis:** Evaluates performance under calm vs. stressed volatility regimes.  

---

## 🏗️ Architecture
Data → Greeks Engine → Hedging Policy → Execution Simulator → Metrics & Reports


- **Data Layer:** Ingests raw options chains and underlying tick data.  
- **Model Layer:** Computes sensitivities and risk exposures.  
- **Policy Layer:** Chooses hedge actions based on config.  
- **Execution Layer:** Simulates order placement and fills.  
- **Analytics Layer:** Produces risk reports, attribution studies, and MLflow‑tracked experiments.  

---

## 📊 Example Use Cases
- **Academic:** Demonstrates mastery of stochastic calculus, volatility modeling, and market microstructure.  
- **Recruiter‑Facing:** Shows ability to design **end‑to‑end trading infrastructure** with professional documentation.  
- **Desk‑Facing:** Prototype for auto‑hedging workflows that could be extended to production systems.  

---

## 🔬 Research Highlights
- **Volatility Regimes:** Backtests across 2008 crisis, 2020 COVID crash, and low‑volatility periods.  
- **Policy Comparison:** Delta‑neutral vs. gamma‑scaled hedging efficiency under different liquidity conditions.  
- **Execution Sensitivity:** Impact of spread widening and order book depth on hedging costs.  

---

## ⚙️ Reproducibility
- **MLflow Tracking:** Every run logs configs, metrics, and artifacts.  
- **Config‑Driven:** YAML files define models, policies, and execution environments.  
- **Audit‑Ready:** Explicit data type management and schema validation.  

---

## 🚀 Getting Started
```bash
# Create environment
make install

# Run a delta‑neutral backtest
make backtest

# Optimize gamma‑scaled policy
make optimize

# Run tests and type checks
make test
make typecheck

## 📚 Documentation
docs/architecture.md – System design and module interactions.

docs/api_reference.md – Function/class documentation.

docs/research_notes.md – Theoretical background and references.

## 🧠 Expertise Demonstrated
Options Theory: Greeks, stochastic volatility, hedging strategies.

Algorithmic Trading: Execution cost modeling, market microstructure simulation.

Quant Engineering: Modular pipelines, MLflow tracking, reproducible configs.

Professional Standards: Testing, type safety, documentation, and CI‑ready workflows.

##📌 Author
Kenneth Quantitative Researcher | Applied Mathematics & Computer Science 
Focus: Systematic trading, volatility modeling, and reproducible quant infrastructure.
