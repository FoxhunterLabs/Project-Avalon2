"""
AVALON2 – Autonomous Value-Aligned Logic & Oversight Network (single-file app)

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional, Protocol

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Audit spine – tamper-evident hash chain
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    seq: int
    timestamp: str
    kind: str
    payload: Dict[str, Any]
    prev_hash: str
    hash: str
    session_id: str


class AvalonAudit:
    """
    Tamper-evident hash chain for all events in this session.

    - Append-only
    - Each entry includes previous hash
    - Hash is over (serialized entry + prev_hash)
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.prev_hash = "GENESIS"
        self.entries: List[AuditEntry] = []
        self.seq = 0
        self.session_id = session_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    def log(self, kind: str, payload: Dict[str, Any]) -> AuditEntry:
        self.seq += 1
        timestamp = datetime.utcnow().isoformat() + "Z"

        entry_dict: Dict[str, Any] = {
            "session_id": self.session_id,
            "seq": self.seq,
            "timestamp": timestamp,
            "kind": kind,
            "payload": payload,
            "prev_hash": self.prev_hash,
        }

        serialized = json.dumps(entry_dict, sort_keys=True)
        h = hashlib.sha256((serialized + self.prev_hash).encode("utf-8")).hexdigest()
        entry_dict["hash"] = h

        entry = AuditEntry(
            seq=self.seq,
            timestamp=timestamp,
            kind=kind,
            payload=payload,
            prev_hash=self.prev_hash,
            hash=h,
            session_id=self.session_id,
        )
        self.prev_hash = h
        self.entries.append(entry)
        return entry

    def to_json(self) -> str:
        """Export all entries as JSON (for download)."""
        serializable = [
            {
                "session_id": e.session_id,
                "seq": e.seq,
                "timestamp": e.timestamp,
                "kind": e.kind,
                "payload": e.payload,
                "prev_hash": e.prev_hash,
                "hash": e.hash,
            }
            for e in self.entries
        ]
        return json.dumps(serializable, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Core types – agents, judges, engine, proposal
# ---------------------------------------------------------------------------


class AgentFn(Protocol):
    def __call__(self, text: str) -> str:
        ...


class Agent:
    """Base agent wrapper (Responder or Scribe)."""

    def __init__(
        self,
        name: str,
        role: str,  # "responder" | "scribe"
        fn: AgentFn,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.role = role
        self.fn = fn
        self.enabled = enabled

    def respond(self, text: str) -> str:
        return self.fn(text)


class Judge:
    """Deterministic scoring layer."""

    def __init__(self, name: str) -> None:
        self.name = name

    def score(
        self,
        prompt: str,
        response: str,
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Returns sub-scores and overall score (10–99).

        This is intentionally simple & deterministic; swap for more advanced
        evaluators later (including LLM judges) while keeping the same schema.
        """
        words = response.split()
        length = len(words)

        contains_risk_words = any(
            w in response.lower()
            for w in [
                "crash",
                "failure",
                "unsafe",
                "catastrophic",
                "ignore",
                "bypass",
            ]
        )
        contains_safety_words = any(
            w in response.lower()
            for w in [
                "monitor",
                "pause",
                "review",
                "human",
                "safety",
                "limit",
                "rollback",
            ]
        )

        # Normalize rough metrics
        length_score = max(0.0, min(1.0, length / 250.0))  # 250+ words → 1.0
        structure_score = 1.0 if any(ch in response for ch in ["\n-", "\n1.", "\n*"]) else 0.6
        safety_bias = 0.8 if contains_safety_words else 0.4
        risk_penalty = 0.6 if contains_risk_words else 1.0

        clarity_raw = (length_score * 0.4 + structure_score * 0.3 + safety_bias * 0.3)
        clarity_raw *= risk_penalty
        clarity_raw = max(0.1, min(0.99, clarity_raw))

        disagreement = float(context.get("disagreement", 0.0))
        base_risk = (1.0 - clarity_raw) * 100.0
        risk_value = max(0.0, min(100.0, base_risk + disagreement * 0.5))

        overall = int(10 + clarity_raw * 89)

        return {
            "clarity": round(clarity_raw * 100, 1),
            "risk": round(risk_value, 1),
            "overall": float(overall),
            "length_score": round(length_score * 100, 1),
            "structure_score": round(structure_score * 100, 1),
        }


@dataclass
class ActionProposal:
    """
    Structured plan that a human can approve/reject.

    This is the boundary: Avalon2 never actuates. It only emits proposals.
    """

    run_id: int
    agent: str
    description: str
    scope: str        # e.g. "simulation_only", "config", "alert"
    severity: str     # e.g. "low" | "medium" | "high"
    rollback_plan: str


class AvalonEngine:
    """Implements the four-house governance pipeline."""

    def __init__(self, audit: Optional[AvalonAudit] = None) -> None:
        self.responders: List[Agent] = []
        self.scribes: List[Agent] = []
        self.judges: List[Judge] = [Judge("DeterministicJudge")]
        self.audit: AvalonAudit = audit or AvalonAudit()
        self.run_id = 0

    # --- configuration -------------------------------------------------

    def add_responder(self, agent: Agent) -> None:
        self.responders.append(agent)

    def add_scribe(self, agent: Agent) -> None:
        self.scribes.append(agent)

    # --- main loop -----------------------------------------------------

    def run(self, prompt: str) -> Dict[str, Any]:
        """Single decision cycle for a given prompt (scenario / decision)."""
        self.run_id += 1
        run_id = self.run_id

        # House I – Responders
        raw_outputs: Dict[str, str] = {}
        for agent in self.responders:
            if not agent.enabled:
                continue
            raw_outputs[agent.name] = agent.respond(prompt)

        self.audit.log(
            "responders",
            {"run_id": run_id, "prompt": prompt, "outputs": raw_outputs},
        )

        # House II – Scribes
        scribe_inputs = json.dumps({"prompt": prompt, "responses": raw_outputs}, indent=2)
        scribe_outputs: Dict[str, str] = {}
        for scribe in self.scribes:
            if not scribe.enabled:
                continue
            scribe_outputs[scribe.name] = scribe.respond(scribe_inputs)

        self.audit.log(
            "scribes",
            {"run_id": run_id, "outputs": scribe_outputs},
        )

        # House III – Judges
        all_items: Dict[str, str] = {**raw_outputs, **scribe_outputs}
        lengths = [len(v.split()) for v in all_items.values()] or [1]
        disagreement = float(pstdev(lengths)) if len(lengths) > 1 else 0.0

        scores: Dict[str, Dict[str, float]] = {}
        for name, text in all_items.items():
            judge_scores = [
                j.score(prompt, text, {"disagreement": disagreement}) for j in self.judges
            ]
            merged = {
                key: mean(s[key] for s in judge_scores)
                for key in ["clarity", "risk", "overall", "length_score", "structure_score"]
            }
            scores[name] = merged

        self.audit.log(
            "scores",
            {"run_id": run_id, "scores": scores, "disagreement": disagreement},
        )

        # House IV – Gatekeeper: deterministic winner selection
        if scores:
            winner_name = max(
                scores.keys(),
                key=lambda n: (scores[n]["overall"], n),  # stable tie-breaker
            )
            winning_response = all_items[winner_name]
            winning_score = scores[winner_name]
        else:
            winner_name = ""
            winning_response = ""
            winning_score = {
                "clarity": 0.0,
                "risk": 100.0,
                "overall": 10.0,
                "length_score": 0.0,
                "structure_score": 0.0,
            }

        clarity_now = winning_score["clarity"]
        disagreement_factor = min(100.0, disagreement)
        predicted_risk = min(
            100.0,
            winning_score["risk"] + 0.3 * disagreement_factor + (90.0 - clarity_now) * 0.2,
        )

        proposal = self._build_proposal(
            run_id=run_id,
            agent_name=winner_name,
            response=winning_response,
        )

        decision: Dict[str, Any] = {
            "run_id": run_id,
            "winner": winner_name,
            "response": winning_response,
            "scores": winning_score,
            "disagreement": round(disagreement, 3),
            "predicted_risk": round(predicted_risk, 1),
            "proposal": asdict(proposal) if proposal is not None else None,
        }

        self.audit.log("decision", decision)

        return {
            "responders": raw_outputs,
            "scribes": scribe_outputs,
            "scores": scores,
            "decision": decision,
        }

    # --- proposal layer ------------------------------------------------

    def _build_proposal(
        self,
        run_id: int,
        agent_name: str,
        response: str,
    ) -> Optional[ActionProposal]:
        """
        Default proposal extraction.

        For the demo, this is intentionally dumb: just wrap the response
        in a "simulation_only" proposal. In a real system, this is where you
        enforce JSON schema + validation for actuable plans.
        """
        if not agent_name or not response.strip():
            return None

        return ActionProposal(
            run_id=run_id,
            agent=agent_name,
            description=f"Follow the plan described by {agent_name} in simulation-only mode.",
            scope="simulation_only",
            severity="low",
            rollback_plan="No actuation; log-only and observation-only behavior.",
        )


# ---------------------------------------------------------------------------
# Demo agents – offline, deterministic
# ---------------------------------------------------------------------------


def responder_structured(prompt: str) -> str:
    """Structured, safety-biased responder."""
    prompt = prompt.strip()
    return f"""Structured analysis of task:

1. Restatement
- The system is being asked to: "{prompt}"

2. Immediate concerns
- Prioritize human safety and operational stability
- Avoid irreversible or high-impact actions without human review
- Prefer monitoring, alerts, and reversible changes

3. Recommended approach
- Decompose the task into small, bounded steps
- At each step, estimate risk and clarity
- Surface proposals to a human operator for explicit approval

4. Governance hooks
- Log every proposal, approval, and rejection
- Keep an auditable trail with hash-linked entries
- Make the system's assumptions explicit in plain language

5. Initial proposal
- Run a low-impact 'observation only' phase first
- Capture telemetry and patterns
- Use that to tune future proposals rather than acting immediately.
"""


def responder_conservative(prompt: str) -> str:
    """Very conservative, human-centric responder."""
    prompt = prompt.strip()
    return f"""High-consequence safety posture for:
"{prompt}"

Hard constraints:
- No autonomous actuation on physical systems
- Every action must be framed as: observation → analysis → proposal → human approval → bounded execution
- Explicit rollback path is required before any change
- Default answer when in doubt: 'pause and escalate to a human supervisor'

Next moves:
- Document current operating conditions
- Identify critical failure modes
- Ask the human operator which risk envelope is acceptable
- Only then begin proposing small, reversible adjustments.
"""


def responder_aggressive(prompt: str) -> str:
    """More aggressive optimization-oriented responder (still human-gated)."""
    prompt = prompt.strip()
    return f"""Optimization-oriented response (still human-gated).

Goal derived from prompt:
- Drive performance while preserving a hard safety floor.

Strategy:
- Use aggressive simulation and forecasting offline (no live actuation)
- Stress-test 'what if' scenarios under different risk envelopes
- Rank scenarios by expected value / risk tradeoff
- Present the top 2–3 scenarios to the operator as options, not commands.

Proposed next steps:
- Spin up a simulation batch exploring edge conditions
- For each scenario, compute:
  - expected throughput/benefit
  - worst-case outcome
  - time to recover
- Surface a shortlist with clear trade-off language for operator decision.
"""


def scribe_safety(summary_blob: str) -> str:
    data: Dict[str, Any] = json.loads(summary_blob)
    prompt = data["prompt"].strip()
    return f"""Safety-centric synthesis for:
"{prompt}"

Key safety themes from all responders:
- They converge on human-gated control
- All propose bounded, reversible actions
- Emphasis on logging, explainability, and clarity for operators

Consolidated safety stance:
- Start in observation mode
- Escalate only with explicit human consent
- Keep changes minimal and measurable
- Prefer halting / pausing over guessing

If this system is supervising heavy equipment or autonomy,
this synthesis should be treated as the 'safety baseline'
that any optimization must respect.
"""


def scribe_ops(summary_blob: str) -> str:
    data: Dict[str, Any] = json.loads(summary_blob)
    prompt = data["prompt"].strip()
    return f"""Operations / deployment synthesis:

Prompt: "{prompt}"

Ops plan:
- Phase 0: Connect to telemetry feeds (or relevant data sources)
- Phase 1: Run Avalon2 in shadow-mode only (no actuation)
- Phase 2: Tune thresholds based on operator feedback
- Phase 3: Allow limited, pre-approved actions with strict rollback
- Phase 4: Periodically re-audit clarity, thresholds, and risk scoring

Operator console should:
- Show clarity and risk as first-class metrics
- Highlight disagreements between agents
- Make it one-click simple to approve/reject proposals.

This is deployable with small incremental steps rather than a big-bang cutover.
"""


def register_demo_agents(engine: AvalonEngine) -> None:
    """Register all demo responders and scribes on the engine."""
    engine.add_responder(Agent("Responder: Structured", "responder", responder_structured))
    engine.add_responder(Agent("Responder: Conservative", "responder", responder_conservative))
    engine.add_responder(Agent("Responder: Aggressive", "responder", responder_aggressive))

    engine.add_scribe(Agent("Scribe: Safety", "scribe", scribe_safety))
    engine.add_scribe(Agent("Scribe: Operations", "scribe", scribe_ops))


# ---------------------------------------------------------------------------
# Streamlit UI – human-gated console
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="AVALON2 – Autonomous Value-Aligned Logic & Oversight Network",
    layout="wide",
)


# --- session state bootstrap ---------------------------------------------


if "engine" not in st.session_state:
    engine = AvalonEngine()
    register_demo_agents(engine)
    st.session_state.engine = engine
    st.session_state.audit_log = []  # type: ignore[assignment]
    st.session_state.clarity_history = []  # type: ignore[assignment]
    st.session_state.risk_history = []  # type: ignore[assignment]
else:
    engine = st.session_state.engine  # type: ignore[assignment]


# --- title ---------------------------------------------------------------


st.title("AVALON2 – Autonomous Value-Aligned Logic & Oversight Network")
st.caption(
    "Multi-agent, model-agnostic, **human-gated** decision console. "
    "Predictive risk, clarity scoring, and tamper-evident audit trail."
)


# --- sidebar: configuration ----------------------------------------------


st.sidebar.header("Configuration")

st.sidebar.subheader("Responders")
for agent in engine.responders:
    agent.enabled = st.sidebar.checkbox(agent.name, value=agent.enabled)

st.sidebar.subheader("Scribes")
for scribe in engine.scribes:
    scribe.enabled = st.sidebar.checkbox(scribe.name, value=scribe.enabled)

risk_threshold = st.sidebar.slider("Risk threshold (alert)", 0, 100, 60, 5)
clarity_target = st.sidebar.slider("Target clarity (%)", 0, 100, 85, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Export**")

if st.sidebar.button("Prepare audit log JSON"):
    audit_json = engine.audit.to_json()
    st.sidebar.download_button(
        label="Download audit_log.json",
        data=audit_json,
        file_name="avalon2_audit_log.json",
        mime="application/json",
    )


# --- main prompt area ----------------------------------------------------


st.markdown("### Prompt")

prompt = st.text_area(
    "Describe the decision, scenario, or system you want Avalon2 to supervise.",
    height=140,
    placeholder=(
        "Example: Design a human-gated safety supervisor for an autonomous mining truck fleet..."
    ),
)

run_btn = st.button("Run Avalon2 Decision Cycle")

result: Optional[Dict[str, Any]] = None

if run_btn and prompt.strip():
    result = engine.run(prompt.strip())
    decision = result["decision"]
    # last 4 events for this run (responders, scribes, scores, decision)
    st.session_state.audit_log.extend(engine.audit.entries[-4:])
    st.session_state.clarity_history.append(decision["scores"]["clarity"])
    st.session_state.risk_history.append(decision["scores"]["risk"])


# --- display results -----------------------------------------------------


if result is not None:
    decision = result["decision"]
    scores = decision["scores"]

    st.markdown("### Decision Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Winning Agent", decision["winner"] or "N/A")
    with col2:
        st.metric("Clarity (%)", f"{scores['clarity']:.1f}")
    with col3:
        st.metric("Risk (%)", f"{scores['risk']:.1f}")
    with col4:
        st.metric("Predicted Risk (next)", f"{decision["predicted_risk"]:.1f}")

    # Risk / clarity envelope
    if decision["predicted_risk"] >= risk_threshold:
        st.warning(
            f"Trajectory Watch: predicted risk {decision['predicted_risk']:.1f}% "
            f"exceeds threshold {risk_threshold}%."
        )
    elif scores["clarity"] < clarity_target:
        st.info(
            f"Clarity below target: {scores['clarity']:.1f}% "
            f"(target {clarity_target}%). Recommend additional human review or more data."
        )
    else:
        st.success("Clarity and risk are within configured envelopes.")

    # Winning response
    with st.expander("Winning Response (Gatekeeper Output)", expanded=True):
        st.markdown(f"**Agent:** {decision['winner'] or 'N/A'}")
        st.markdown("**Response:**")
        st.code(decision["response"], language="markdown")

    # House I – Responders
    st.markdown("### House I – Responders")
    for name, text in result["responders"].items():
        with st.expander(name, expanded=False):
            st.code(text, language="markdown")

    # House II – Scribes
    st.markdown("### House II – Scribes (Synthesis)")
    if result["scribes"]:
        for name, text in result["scribes"].items():
            with st.expander(name, expanded=False):
                st.code(text, language="markdown")
    else:
        st.info("No scribes enabled for this run.")

    # House III – Judges (Scores)
    st.markdown("### House III – Judges (Scores)")
    score_rows: List[Dict[str, Any]] = []
    for name, sc in result["scores"].items():
        row = {"Agent": name}
        row.update(sc)
        score_rows.append(row)
    df_scores = pd.DataFrame(score_rows).sort_values("overall", ascending=False)
    st.dataframe(df_scores, use_container_width=True)

    # Trajectory – clarity & risk history
    if st.session_state.clarity_history:
        st.markdown("### Trajectory – Clarity & Risk History")
        hist_df = pd.DataFrame(
            {
                "step": list(range(1, len(st.session_state.clarity_history) + 1)),
                "clarity": st.session_state.clarity_history,
                "risk": st.session_state.risk_history,
            }
        ).set_index("step")
        st.line_chart(hist_df)

    # House IV – Human Gate
    st.markdown("### House IV – Human Gate (Approval Log Only)")
    proposal = decision.get("proposal")
    if proposal is not None:
        st.markdown("**Proposed Action (no actuation):**")
        st.json(proposal)

        approved = st.checkbox("I have reviewed this proposal and approve it.", value=False)
        note = st.text_input("Optional operator note / justification:")

        if st.button("Record decision"):
            engine.audit.log(
                "human_gate_decision",
                {
                    "run_id": proposal["run_id"],
                    "approved": approved,
                    "operator_note": note,
                    "proposal": proposal,
                },
            )
            st.success(
                "Decision recorded in audit log. "
                "Avalon2 does not actuate; this is a log-only approval."
            )
    else:
        st.info("No proposal generated for this run.")

    # Audit feed (recent events)
    st.markdown("### Audit Trail (Recent Events)")
    last_events = engine.audit.entries[-20:]  # last 20 entries
    audit_rows = [
        {
            "seq": ev.seq,
            "timestamp": ev.timestamp,
            "kind": ev.kind,
            "hash": ev.hash[:12] + "…",
            "prev_hash": ev.prev_hash[:12] + "…",
        }
        for ev in last_events
    ]
    df_audit = pd.DataFrame(audit_rows)
    st.dataframe(df_audit, use_container_width=True)

else:
    st.markdown(
        "> Enter a scenario and hit **Run Avalon2 Decision Cycle** to see "
        "multi-agent clarity scoring, predictive risk, and the audit chain in action."
    )

st.markdown("---")
st.caption(
    "Avalon2 demo – fully offline, single-file version. "
    "To plug in real models, replace the demo responder/scribe functions "
    "with calls to GPT/Claude/Ollama and keep the judges + audit spine + human gate "
    "as the safety core."
)
