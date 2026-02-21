# SQLTranslator — Pharma R&D Data Intelligence Platform

**Portfolio Project by Jon Tziv** | Natural language → SQL → Insights, purpose-built for pharmaceutical R&D

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Built on Vanna AI](https://img.shields.io/badge/built%20on-Vanna%20AI-purple.svg)](https://vanna.ai)
[![Domain](https://img.shields.io/badge/domain-Pharma%20R%26D-green.svg)]()

---

## Portfolio Context

This project demonstrates rapid AI product prototyping applied to a high-value pharmaceutical R&D use case. It is built as a minimum viable product (MVP) showcasing the kind of work I do as a product leader at the intersection of AI, data, and regulated-industry delivery.

**Skills demonstrated:**
- **AI Product Vision** — Identified a concrete, high-ROI problem (R&D analysts spending hours writing SQL) and scoped a focused MVP
- **Rapid Prototyping** — Extended an open-source AI framework (Vanna AI) with pharma-specific tooling in days, not quarters
- **User-Centered Design** — Conversation flows designed around real R&D analyst workflows (pipeline queries, compound data, trial status)
- **Enterprise Readiness** — Row-level security, audit logging, and permission-gated SQL visibility to meet regulated-industry compliance needs
- **Lean Roadmap Thinking** — MVP scoped to highest-value features; clear backlog articulates future phases without over-engineering now

**Relevance to the FIT Product Manager role:** This project mirrors exactly what Forward Impact Engineering teams do — take a validated open-source foundation, add domain-specific intelligence, and ship a working prototype that proves business value before committing to full-scale investment.

---

## The Problem

Pharmaceutical R&D teams generate enormous volumes of structured data — compound libraries, clinical trial results, pipeline stage metrics, regulatory submission timelines. Yet most analysts spend significant time writing and debugging SQL queries rather than interpreting results.

> *"How many Phase II compounds advanced to Phase III in the last 12 months?"*
> *"Which therapeutic areas have the highest attrition rate at Phase II?"*
> *"Show me all trials for compound X with patient enrollment above 500."*

These questions have clear answers in existing databases. The bottleneck is translation — from business question to SQL.

**SQLTranslator removes that bottleneck.**

---

## MVP Scope (What's Built)

| Feature | Status | Notes |
|---|---|---|
| Natural language → SQL | ✅ | Powered by any LLM (Claude, GPT, Gemini, Ollama) |
| Pharma R&D demo database | ✅ | Compounds, trials, pipeline stages, therapeutic areas |
| Streaming chat interface | ✅ | Real-time tables, charts, SQL, narrative summary |
| Role-based SQL visibility | ✅ | SQL shown only to `admin` users; analysts see results only |
| Audit logging | ✅ | Every query tracked per user for compliance |
| Web component embed | ✅ | `<sqltranslator-chat>` drops into any existing web app |
| FastAPI / Flask server | ✅ | Production-ready backend, bring your own auth |
| Mock mode (no API key) | ✅ | Full demo runnable without any LLM credentials |

---

## Architecture

```
User Question (natural language)
        │
        ▼
┌──────────────────────────────────┐
│     SQLTranslator Agent          │
│  (user-aware, permission-gated)  │
└──────────┬───────────────────────┘
           │
    ┌──────▼──────┐    ┌──────────────────┐
    │  LLM Layer  │    │   Tool Registry  │
    │  (any LLM)  │    │  RunSql | Chart  │
    └──────┬──────┘    │  Memory | Python │
           │           └──────────────────┘
           ▼
┌──────────────────────┐
│   R&D Database       │
│  (Snowflake/PG/      │
│   SQLite/BigQuery)   │
└──────────────────────┘
           │
           ▼
   Streaming Response:
   SQL → Table → Chart → Summary
```

The agent is **user-aware at every layer** — identity flows through system prompts, tool execution, and SQL filtering so row-level security is enforced automatically.

---

## Pharma R&D Demo — Quickstart

### Option 1: No API Key Required (Mock Mode)

```bash
git clone <this-repo>
cd vanna-main
pip install -e .

# Run the pharma R&D demo with mock LLM
python -m vanna.examples.pharma_rnd_demo
```

This runs a complete demonstration against a local SQLite database pre-loaded with mock pharma R&D data — compounds, clinical trials, pipeline stages, and therapeutic area metrics.

**Sample output:**
```
SQLTranslator — Pharma R&D Intelligence Demo
============================================
Database: pharma_rnd.db (compounds, trials, pipeline, therapeutic areas)

Query 1: "How many compounds are currently in Phase II trials?"
[Thinking...] Generating SQL...

SQL:
  SELECT COUNT(*) as compound_count
  FROM clinical_trials
  WHERE phase = 'Phase II' AND status = 'Active';

Result:
  compound_count
  --------------
  23

Summary: There are currently 23 compounds in active Phase II trials.

Query 2: "Which therapeutic areas have the most pipeline compounds?"
...
```

### Option 2: With Real LLM (Claude / OpenAI)

```bash
pip install -e ".[anthropic,fastapi]"
export ANTHROPIC_API_KEY=your_key_here

python -m vanna.examples.pharma_rnd_demo --llm anthropic
```

### Option 3: Full Web UI

```bash
pip install -e ".[anthropic,fastapi]"
export ANTHROPIC_API_KEY=your_key_here

python -m vanna.servers --example pharma_rnd_demo
# Open http://localhost:8000
```

---

## R&D Analytics Use Cases

The demo database supports queries across the full drug development pipeline:

**Pipeline Intelligence**
```
"How many compounds advanced from Phase II to Phase III in 2024?"
"What's the attrition rate at each clinical phase for oncology compounds?"
"Show me the top 10 compounds by projected peak sales."
```

**Clinical Trial Monitoring**
```
"List all active Phase III trials with enrollment below target."
"Which trials have had serious adverse events in the last 6 months?"
"Show enrollment velocity for trial TRIAL-2024-017."
```

**Therapeutic Area Analytics**
```
"Compare R&D spend vs. pipeline output by therapeutic area."
"Which indication has the fastest Phase II to Phase III transition time?"
"Rank therapeutic areas by number of first-in-class compounds."
```

**Regulatory & Milestone Tracking**
```
"Which compounds have NDA submissions due in Q3 2025?"
"Show all compounds that missed their last milestone by more than 90 days."
"List compounds awaiting FDA breakthrough therapy designation."
```

---

## Technical Implementation

### Stack

| Layer | Technology |
|---|---|
| Core AI Agent | Vanna AI 2.0 framework |
| LLM | Claude (Anthropic), GPT-4 (OpenAI), or any Ollama model |
| Default Demo DB | SQLite (pharma_rnd.db) |
| Production DB | Snowflake, BigQuery, PostgreSQL, Redshift |
| Server | FastAPI + SSE streaming |
| Frontend | `<sqltranslator-chat>` web component |
| Auth | Bring your own (cookies, JWT, OAuth) |
| Security | Row-level filtering, per-user audit logs |

### User-Aware Security Model

```python
from vanna import Agent, User
from vanna.core.user import UserResolver, RequestContext

class PharmaUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        token = request_context.get_header('Authorization')
        user_data = self.verify_token(token)  # Your existing SSO/auth

        return User(
            id=user_data['employee_id'],
            email=user_data['email'],
            group_memberships=user_data['groups']  # e.g. ['oncology_analysts', 'phase2_data']
        )
```

Row-level security is applied automatically — an oncology analyst only sees oncology trial data, even if they ask a broad question.

### Extending with Custom Tools

```python
from vanna.core.tool import Tool, ToolContext, ToolResult
from pydantic import BaseModel, Field

class CompoundLookupArgs(BaseModel):
    compound_id: str = Field(description="The compound identifier (e.g. CPD-2024-001)")

class CompoundLookupTool(Tool[CompoundLookupArgs]):
    """Fetch full compound profile from the molecular registry."""

    @property
    def name(self) -> str:
        return "lookup_compound"

    @property
    def access_groups(self) -> list[str]:
        return ["compound_registry_read"]

    def get_args_schema(self):
        return CompoundLookupArgs

    async def execute(self, context: ToolContext, args: CompoundLookupArgs) -> ToolResult:
        compound = await self.registry_client.get(args.compound_id)
        return ToolResult(success=True, result_for_llm=compound.to_summary())
```

---

## Product Roadmap

This MVP validates the core hypothesis: *R&D analysts can self-serve data queries without SQL knowledge.*

### Phase 1 — MVP (Current)
- [x] Natural language to SQL with pharma R&D demo data
- [x] Streaming web interface
- [x] Role-based access (analyst vs. admin views)
- [x] Audit logging for GxP compliance readiness

### Phase 2 — Validated Product
- [ ] Integration with Veeva Vault / CTMS data sources
- [ ] Pre-trained schema understanding for common pharma data models (CDISC SDTM, ADaM)
- [ ] Saved queries and scheduled report generation
- [ ] Slack / Teams bot integration for ad-hoc pipeline queries

### Phase 3 — Enterprise Scale
- [ ] Multi-tenant deployment (each therapeutic area team isolated)
- [ ] Fine-tuned domain model with pharma-specific SQL patterns
- [ ] Anomaly detection alerts ("enrollment dropped 30% this week")
- [ ] Integration with R (for biostatisticians) alongside Python/SQL

---

## Why This Approach

### PM Rationale for the MVP Scope

I scoped this MVP around three principles:

1. **Fastest path to value**: Natural language → SQL has immediate ROI for analysts. No workflow change required — they just ask questions.
2. **Compliance first**: In pharma, security and audit requirements are non-negotiable. Building row-level security and audit logging into the MVP (not as an afterthought) means no rework when scaling.
3. **Extensible foundation**: By building on the Vanna AI framework rather than from scratch, the MVP is delivered faster and benefits from community-maintained LLM and database integrations — exactly the kind of build-vs-buy decision a PM should make.

---

## Running the Tests

```bash
pip install -e ".[test]"
pytest tests/ -v
```

---

## About This Project

**SQLTranslator** is a portfolio project by **Jon Tziv** demonstrating rapid AI product prototyping for enterprise pharma use cases.

Built on the [Vanna AI](https://github.com/vanna-ai/vanna) open-source framework (MIT License), with pharma-specific extensions, demo data, and product framing by Jon Tziv.

**Contact:** Available via LinkedIn | GitHub

---

*Built to demonstrate what's possible when product thinking meets AI infrastructure — fast.*
