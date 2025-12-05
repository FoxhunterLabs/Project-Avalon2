AVALON2 – Autonomous Value-Aligned Logic & Oversight Network

AVALON2 is a **model-agnostic, human-gated decision console** with:

- Multi-agent responders and scribes
- Deterministic scoring/judging (clarity, risk, structure)
- A tamper-evident audit chain
- Structured **ActionProposal** objects instead of direct actuation

It’s designed as an oversight / safety console that can sit **on top of** autonomy, infra, or ops systems. AVALON2 **never actuates**; it emits proposals and logs human decisions.

---

## Features

- **Human-gated by design**
  - No direct calls to actuators
  - Outputs structured proposals for operators to approve/reject
- **Deterministic core**
  - Fixed scoring logic, stable tie-breaking
  - No randomness in selection logic
- **Tamper-evident audit spine**
  - Hash-chained audit entries with `prev_hash`
  - Exportable as JSON for offline review
- **Multi-agent governance**
  - Responders: generate candidate plans
  - Scribes: synthesize / summarize plans
  - Judges: score clarity & risk
  - Gatekeeper: picks a winner and builds a proposal
- **Single-file deployment**
  - Everything lives in `app.py`
  - Easy to read, fork, and extend

---

## Quickstart

```bash
# Clone your fork, then:

pip install -r requirements.txt
streamlit run app.py
Open the URL Streamlit prints (usually http://localhost:8501).
________________________________________
How to Use
1.	Describe a scenario
In the Prompt box, describe the decision / system you want overseen
(e.g., “Shadow supervisor for autonomous haul trucks” or “Ops console for datacenter cooling”).
2.	Configure agents (sidebar)
o	Enable/disable responder agents
o	Enable/disable scribe agents
o	Set your risk threshold and clarity target
3.	Run a decision cycle
Click “Run Avalon2 Decision Cycle”. AVALON2 will:
o	Run all enabled responders
o	Run all enabled scribes
o	Score all outputs deterministically
o	Select a winning agent and compute predicted next-step risk
o	Generate a structured ActionProposal (simulation-only by default)
4.	Review & log human decision
o	Inspect the winning response and all agent outputs
o	Review the proposal in House IV – Human Gate
o	Approve or reject and record your decision (logged in the audit chain ONLY)
AVALON2 does not actuate. It only logs proposals and operator decisions.
________________________________________
Architecture (Single-File Overview)
All of this lives in app.py:
•	AvalonAudit
o	Hash-chained audit log with seq, timestamp, kind, payload, prev_hash, hash
o	to_json() for exporting a full session
•	Agent
o	Thin wrapper around a callable fn(text: str) -> str
o	Used for both responders and scribes
•	Judge
o	Deterministic scoring using:
	Length / structure heuristics
	Simple safety/risk keywords
	Disagreement across agents (response-length variance)
•	AvalonEngine
o	Implements the four houses:
1.	Responders → raw outputs
2.	Scribes → synthesis over all outputs
3.	Judges → scores for each agent
4.	Gatekeeper → deterministic winner + ActionProposal
o	Logs each stage into AvalonAudit
•	Demo agents (offline, no LLMs):
o	responder_structured, responder_conservative, responder_aggressive
o	scribe_safety, scribe_ops
•	Streamlit UI:
o	Sidebar configuration
o	Decision snapshot & history chart
o	Responders, scribes, scores
o	Human Gate panel to record approvals
o	Audit trail table + JSON export
________________________________________
Plugging in Real Models (LLMs, etc.)
This repo ships fully offline. To integrate real models:
•	Replace a demo responder with a model call:
•	def responder_llm(prompt: str) -> str:
•	    # TODO: plug in GPT/Claude/Ollama/etc.
•	    # Must return plain text; Avalon2 treats it as untrusted proposal text.
•	    ...
•	Keep the core logic unchanged:
o	Do not let models set thresholds or bypass the gate
o	Treat model outputs as untrusted advice that flows through:
	judges → scores → ActionProposal → human gate
For anything high-consequence, you should still have a separate, non-LLM safety layer that enforces hard physical/operational rules before any actuation.
________________________________________
License
 MIT
