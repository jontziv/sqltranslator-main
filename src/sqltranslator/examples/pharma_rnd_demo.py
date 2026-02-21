"""
SQLTranslator — Pharma R&D Pipeline Intelligence Demo
Portfolio project by Jon Tziv

Demonstrates natural language to SQL for pharmaceutical R&D data:
- Drug pipeline analytics (compounds by phase, attrition rates)
- Clinical trial monitoring (enrollment, adverse events, milestones)
- Therapeutic area analytics (spend vs. output, time-to-phase)
- Regulatory tracking (NDA timelines, milestone status)

Usage:
  No API key needed (mock mode):
    python -m sqltranslator.examples.pharma_rnd_demo

  With real LLM (set ANTHROPIC_API_KEY):
    python -m sqltranslator.examples.pharma_rnd_demo --llm anthropic

  As server:
    python -m sqltranslator.servers --example pharma_rnd_demo
"""

import asyncio
import os
import sqlite3
import uuid
from typing import AsyncGenerator, List, Optional

from sqltranslator import (
    Agent,
    AgentConfig,
    ToolRegistry,
    User,
)
from sqltranslator.core.llm.base import LlmService
from sqltranslator.core.llm.models import LlmRequest, LlmResponse, LlmStreamChunk
from sqltranslator.core.tool.models import ToolCall, ToolSchema
from sqltranslator.core.user.models import User  # noqa: F811 – same class, explicit import
from sqltranslator.core.user.request_context import RequestContext
from sqltranslator.core.user.resolver import UserResolver
from sqltranslator.integrations.local.agent_memory.in_memory import DemoAgentMemory
from sqltranslator.integrations.sqlite import SqliteRunner
from sqltranslator.tools import RunSqlTool

# ---------------------------------------------------------------------------
# Pharma R&D demo database builder
# ---------------------------------------------------------------------------

_PHARMA_SCHEMA = """
CREATE TABLE IF NOT EXISTS compounds (
    compound_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    therapeutic_area TEXT NOT NULL,
    mechanism       TEXT,
    indication      TEXT,
    current_phase   TEXT NOT NULL,
    status          TEXT NOT NULL,
    first_in_class  INTEGER DEFAULT 0,
    projected_peak_sales_m REAL,
    entry_date      TEXT
);

CREATE TABLE IF NOT EXISTS clinical_trials (
    trial_id        TEXT PRIMARY KEY,
    compound_id     TEXT NOT NULL,
    phase           TEXT NOT NULL,
    status          TEXT NOT NULL,
    indication      TEXT,
    sites           INTEGER,
    target_enrollment INTEGER,
    actual_enrollment INTEGER,
    start_date      TEXT,
    primary_completion TEXT,
    serious_adverse_events INTEGER DEFAULT 0,
    FOREIGN KEY (compound_id) REFERENCES compounds(compound_id)
);

CREATE TABLE IF NOT EXISTS pipeline_history (
    event_id        TEXT PRIMARY KEY,
    compound_id     TEXT NOT NULL,
    from_phase      TEXT,
    to_phase        TEXT NOT NULL,
    event_date      TEXT NOT NULL,
    outcome         TEXT,
    FOREIGN KEY (compound_id) REFERENCES compounds(compound_id)
);

CREATE TABLE IF NOT EXISTS milestones (
    milestone_id    TEXT PRIMARY KEY,
    compound_id     TEXT NOT NULL,
    milestone_type  TEXT NOT NULL,
    planned_date    TEXT NOT NULL,
    actual_date     TEXT,
    status          TEXT NOT NULL,
    days_variance   INTEGER,
    FOREIGN KEY (compound_id) REFERENCES compounds(compound_id)
);

CREATE TABLE IF NOT EXISTS rd_spend (
    spend_id        TEXT PRIMARY KEY,
    therapeutic_area TEXT NOT NULL,
    fiscal_year     INTEGER NOT NULL,
    phase           TEXT NOT NULL,
    spend_usd_m     REAL NOT NULL
);
"""

_PHARMA_DATA = """
INSERT OR IGNORE INTO compounds VALUES
('CPD-001','Vonastinib','Oncology','EGFR inhibitor','NSCLC','Phase III','Active',1,1200.0,'2019-03-01'),
('CPD-002','Relicovir','Virology','Protease inhibitor','RSV','Phase II','Active',0,450.0,'2020-07-15'),
('CPD-003','Axaflutide','Immunology','IL-17 blocker','Psoriasis','Phase III','Active',0,800.0,'2018-11-20'),
('CPD-004','Duveltinib','Oncology','BTK inhibitor','CLL','Phase II','Active',0,620.0,'2021-02-10'),
('CPD-005','Nerlanapep','Neurology','GLP-1 agonist','Alzheimers','Phase I','Active',1,980.0,'2022-06-01'),
('CPD-006','Palrucitinib','Immunology','JAK1/2 inhibitor','RA','Phase III','Active',0,750.0,'2019-09-14'),
('CPD-007','Zetomivant','Cardiology','PCSK9 inhibitor','HFrEF','Phase II','Active',0,540.0,'2021-04-22'),
('CPD-008','Foravumab','Oncology','PD-L1 antibody','TNBC','Phase II','Active',0,900.0,'2020-12-03'),
('CPD-009','Quilovexin','Virology','Cap-dep endonuclease','Influenza','Phase I','Active',1,320.0,'2022-09-18'),
('CPD-010','Atrevocept','Immunology','BAFF receptor','Lupus','Phase II','Active',0,410.0,'2021-07-30'),
('CPD-011','Brinafovir','Oncology','CDK4/6 inhibitor','HR+ BC','Phase III','Active',0,1100.0,'2018-05-12'),
('CPD-012','Celevastide','Cardiology','LPA inhibitor','ASCVD','Phase II','Active',1,670.0,'2021-01-08'),
('CPD-013','Denavumab','Neurology','LRRK2 inhibitor','Parkinsons','Phase I','Active',1,860.0,'2023-02-14'),
('CPD-014','Etrolizib','Immunology','alpha4beta7','UC','Phase III','Active',0,580.0,'2017-08-25'),
('CPD-015','Ficlatuzumab','Oncology','HGF antibody','GBM','Phase II','Suspended',0,220.0,'2020-03-19'),
('CPD-016','Gevozinib','Cardiology','Aldosterone syn','HFpEF','Phase I','Active',1,440.0,'2022-11-07'),
('CPD-017','Helomivant','Virology','NS5B polymerase','HCV','Preclinical','Active',0,190.0,'2023-05-23'),
('CPD-018','Ilomastat','Oncology','MMP inhibitor','Pancreatic','Phase II','Active',0,310.0,'2021-10-11'),
('CPD-019','Jorinavir','Virology','Integrase inhibitor','HIV','Phase III','Active',0,730.0,'2019-06-02'),
('CPD-020','Keyvatinib','Oncology','FGFR1-3 inhib','Bladder','Phase II','Active',0,480.0,'2020-09-29'),
('CPD-021','Lanabecestat','Neurology','BACE1 inhibitor','Alzheimers','Discontinued','Discontinued',0,0.0,'2017-01-15'),
('CPD-022','Mavacamten','Cardiology','Myosin inhibitor','HCM','Phase III','Active',1,920.0,'2018-04-03'),
('CPD-023','Namilumab','Immunology','GM-CSF','RA','Phase II','Active',0,360.0,'2021-08-17'),
('CPD-024','Olaparib','Oncology','PARP inhibitor','Ovarian','Approved','Approved',1,1800.0,'2015-02-28'),
('CPD-025','Pirtobrutinib','Oncology','BTK inhibitor','MCL','Phase III','Active',0,840.0,'2019-11-22');

INSERT OR IGNORE INTO clinical_trials VALUES
('TRL-001','CPD-001','Phase III','Active','NSCLC',45,620,589,'2022-01-10','2025-06-30',3),
('TRL-002','CPD-002','Phase II','Active','RSV',28,340,298,'2022-08-22','2024-11-15',1),
('TRL-003','CPD-003','Phase III','Active','Psoriasis',52,780,801,'2021-03-14','2024-09-30',2),
('TRL-004','CPD-004','Phase II','Active','CLL',22,280,241,'2022-11-01','2025-03-31',0),
('TRL-005','CPD-005','Phase I','Active','Alzheimers',8,60,54,'2023-04-03','2025-01-31',0),
('TRL-006','CPD-006','Phase III','Active','RA',61,890,923,'2021-06-20','2024-12-31',4),
('TRL-007','CPD-007','Phase II','Active','HFrEF',19,240,187,'2023-01-15','2025-07-31',1),
('TRL-008','CPD-008','Phase II','Active','TNBC',31,380,342,'2022-05-08','2024-10-31',2),
('TRL-009','CPD-011','Phase III','Active','HR+ BC',58,920,944,'2020-09-01','2024-08-31',5),
('TRL-010','CPD-014','Phase III','Active','UC',49,710,678,'2020-11-12','2024-07-31',3),
('TRL-011','CPD-019','Phase III','Active','HIV',67,1100,1089,'2021-02-28','2025-02-28',1),
('TRL-012','CPD-022','Phase III','Active','HCM',43,560,571,'2021-07-19','2025-05-31',2),
('TRL-013','CPD-025','Phase III','Active','MCL',36,480,412,'2022-03-05','2025-09-30',1),
('TRL-014','CPD-010','Phase II','Active','Lupus',24,300,258,'2022-09-14','2025-01-31',0),
('TRL-015','CPD-012','Phase II','Active','ASCVD',26,320,301,'2022-12-01','2025-04-30',0),
('TRL-016','CPD-018','Phase II','Active','Pancreatic',17,200,162,'2023-02-20','2025-08-31',0),
('TRL-017','CPD-020','Phase II','Active','Bladder',21,260,219,'2022-07-11','2024-12-31',1),
('TRL-018','CPD-023','Phase II','Active','RA',18,220,198,'2022-10-03','2025-02-28',0),
('TRL-019','CPD-015','Phase II','Suspended','GBM',0,200,87,'2021-06-15','2024-06-30',7),
('TRL-020','CPD-009','Phase I','Active','Influenza',6,48,43,'2023-07-01','2025-03-31',0);

INSERT OR IGNORE INTO pipeline_history VALUES
('EVT-001','CPD-001','Phase II','Phase III','2022-01-05','Advanced'),
('EVT-002','CPD-003','Phase II','Phase III','2021-03-01','Advanced'),
('EVT-003','CPD-006','Phase II','Phase III','2021-06-10','Advanced'),
('EVT-004','CPD-011','Phase II','Phase III','2020-08-20','Advanced'),
('EVT-005','CPD-014','Phase II','Phase III','2020-11-01','Advanced'),
('EVT-006','CPD-019','Phase II','Phase III','2021-02-15','Advanced'),
('EVT-007','CPD-022','Phase II','Phase III','2021-07-05','Advanced'),
('EVT-008','CPD-025','Phase II','Phase III','2022-02-20','Advanced'),
('EVT-009','CPD-021','Phase II','Discontinued','2020-04-12','Failed - futility'),
('EVT-010','CPD-024','Phase III','Approved','2020-12-18','NDA Approved'),
('EVT-011','CPD-002','Phase I','Phase II','2022-08-01','Advanced'),
('EVT-012','CPD-004','Phase I','Phase II','2022-10-15','Advanced'),
('EVT-013','CPD-007','Phase I','Phase II','2023-01-05','Advanced'),
('EVT-014','CPD-008','Phase I','Phase II','2022-05-01','Advanced'),
('EVT-015','CPD-015','Phase I','Phase II','2021-06-01','Advanced - later suspended');

INSERT OR IGNORE INTO milestones VALUES
('MLS-001','CPD-001','NDA Submission','2025-09-30',NULL,'On Track',-5),
('MLS-002','CPD-003','NDA Submission','2025-03-31',NULL,'At Risk',42),
('MLS-003','CPD-006','NDA Submission','2025-06-30',NULL,'On Track',12),
('MLS-004','CPD-011','Primary Completion','2024-08-31',NULL,'On Track',0),
('MLS-005','CPD-014','NDA Submission','2025-01-31',NULL,'Delayed',89),
('MLS-006','CPD-019','NDA Submission','2025-09-30',NULL,'On Track',-3),
('MLS-007','CPD-022','NDA Submission','2025-11-30',NULL,'On Track',8),
('MLS-008','CPD-025','Primary Completion','2025-09-30',NULL,'At Risk',55),
('MLS-009','CPD-004','Phase II Complete','2025-06-30',NULL,'On Track',-12),
('MLS-010','CPD-008','Phase II Complete','2024-10-31',NULL,'At Risk',30),
('MLS-011','CPD-002','Phase II Complete','2024-11-15',NULL,'On Track',5),
('MLS-012','CPD-007','Phase II Complete','2025-07-31',NULL,'On Track',15),
('MLS-013','CPD-024','Commercial Launch','2021-06-30','2021-05-14','Completed',-47),
('MLS-014','CPD-010','Phase II Complete','2025-01-31',NULL,'On Track',0),
('MLS-015','CPD-021','Phase II Complete','2021-12-31','2020-04-12','Discontinued',NULL);

INSERT OR IGNORE INTO rd_spend VALUES
('SPD-001','Oncology',2022,'Phase I',45.2),
('SPD-002','Oncology',2022,'Phase II',128.7),
('SPD-003','Oncology',2022,'Phase III',312.4),
('SPD-004','Oncology',2023,'Phase I',52.1),
('SPD-005','Oncology',2023,'Phase II',145.3),
('SPD-006','Oncology',2023,'Phase III',398.6),
('SPD-007','Immunology',2022,'Phase I',28.4),
('SPD-008','Immunology',2022,'Phase II',89.2),
('SPD-009','Immunology',2022,'Phase III',198.5),
('SPD-010','Immunology',2023,'Phase I',31.7),
('SPD-011','Immunology',2023,'Phase II',102.8),
('SPD-012','Immunology',2023,'Phase III',241.3),
('SPD-013','Virology',2022,'Phase I',18.9),
('SPD-014','Virology',2022,'Phase II',52.4),
('SPD-015','Virology',2022,'Phase III',124.7),
('SPD-016','Virology',2023,'Phase I',22.3),
('SPD-017','Virology',2023,'Phase II',61.8),
('SPD-018','Virology',2023,'Phase III',138.9),
('SPD-019','Cardiology',2022,'Phase I',21.6),
('SPD-020','Cardiology',2022,'Phase II',67.3),
('SPD-021','Cardiology',2022,'Phase III',158.2),
('SPD-022','Cardiology',2023,'Phase I',24.8),
('SPD-023','Cardiology',2023,'Phase II',78.4),
('SPD-024','Cardiology',2023,'Phase III',187.5),
('SPD-025','Neurology',2022,'Phase I',38.1),
('SPD-026','Neurology',2022,'Phase II',72.6),
('SPD-027','Neurology',2022,'Phase III',0.0),
('SPD-028','Neurology',2023,'Phase I',44.9),
('SPD-029','Neurology',2023,'Phase II',84.2),
('SPD-030','Neurology',2023,'Phase III',0.0);
"""


def build_pharma_db(db_path: str) -> None:
    """Create and populate the pharma R&D SQLite database."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_PHARMA_SCHEMA)
        conn.executescript(_PHARMA_DATA)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Mock LLM service: pre-scripted pharma R&D queries
# ---------------------------------------------------------------------------

_DEMO_QUERIES = [
    (
        "How many compounds are currently in active Phase II trials?",
        "SELECT COUNT(DISTINCT c.compound_id) AS active_phase2_compounds "
        "FROM compounds c "
        "JOIN clinical_trials t ON c.compound_id = t.compound_id "
        "WHERE t.phase = 'Phase II' AND t.status = 'Active';",
    ),
    (
        "Which therapeutic areas have the most pipeline compounds?",
        "SELECT therapeutic_area, COUNT(*) AS compound_count "
        "FROM compounds "
        "WHERE status NOT IN ('Discontinued', 'Approved') "
        "GROUP BY therapeutic_area "
        "ORDER BY compound_count DESC;",
    ),
    (
        "Show me Phase III trials where actual enrollment is below target.",
        "SELECT t.trial_id, c.name AS compound, c.therapeutic_area, "
        "t.target_enrollment, t.actual_enrollment, "
        "ROUND(100.0 * t.actual_enrollment / t.target_enrollment, 1) AS pct_enrolled "
        "FROM clinical_trials t "
        "JOIN compounds c ON t.compound_id = c.compound_id "
        "WHERE t.phase = 'Phase III' AND t.status = 'Active' "
        "AND t.actual_enrollment < t.target_enrollment "
        "ORDER BY pct_enrolled ASC;",
    ),
    (
        "What is the total R&D spend by therapeutic area in 2023?",
        "SELECT therapeutic_area, "
        "ROUND(SUM(spend_usd_m), 1) AS total_spend_usd_m "
        "FROM rd_spend "
        "WHERE fiscal_year = 2023 "
        "GROUP BY therapeutic_area "
        "ORDER BY total_spend_usd_m DESC;",
    ),
    (
        "Which compounds have milestones that are delayed or at risk?",
        "SELECT c.name AS compound, c.therapeutic_area, "
        "m.milestone_type, m.planned_date, m.status, m.days_variance "
        "FROM milestones m "
        "JOIN compounds c ON m.compound_id = c.compound_id "
        "WHERE m.status IN ('Delayed', 'At Risk') "
        "ORDER BY m.days_variance DESC;",
    ),
    (
        "How many first-in-class compounds do we have per therapeutic area?",
        "SELECT therapeutic_area, "
        "COUNT(*) FILTER (WHERE first_in_class = 1) AS first_in_class_count, "
        "COUNT(*) AS total_compounds "
        "FROM compounds "
        "WHERE status NOT IN ('Discontinued') "
        "GROUP BY therapeutic_area "
        "ORDER BY first_in_class_count DESC;",
    ),
]


class PharmaRndMockLlmService(LlmService):
    """
    Mock LLM for SQLTranslator pharma R&D demo.
    Cycles through pre-scripted NL→SQL pairs to demonstrate the product
    without requiring an API key.
    """

    def __init__(self) -> None:
        self._idx = 0

    def _next_query(self) -> tuple[str, str]:
        pair = _DEMO_QUERIES[self._idx % len(_DEMO_QUERIES)]
        self._idx += 1
        return pair

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        await asyncio.sleep(0.05)
        return self._build_response(request)

    async def stream_request(
        self, request: LlmRequest
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        await asyncio.sleep(0.05)
        response = self._build_response(request)
        if response.tool_calls:
            yield LlmStreamChunk(tool_calls=response.tool_calls)
        if response.content is not None:
            yield LlmStreamChunk(
                content=response.content, finish_reason=response.finish_reason
            )
        else:
            yield LlmStreamChunk(finish_reason=response.finish_reason)

    async def validate_tools(self, tools: List[ToolSchema]) -> List[str]:
        return []

    def _build_response(self, request: LlmRequest) -> LlmResponse:
        last = request.messages[-1] if request.messages else None
        if last and last.role == "tool":
            result = last.content or "No data returned."
            return LlmResponse(
                content=(
                    "Here are the results from the R&D pipeline database:\n\n"
                    + result
                    + "\n\n[SQLTranslator — Pharma R&D Intelligence | Portfolio: Jon Tziv]"
                ),
                finish_reason="stop",
                usage={"prompt_tokens": 40, "completion_tokens": 30, "total_tokens": 70},
            )

        question, sql = self._next_query()
        tool_call = ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name="run_sql",
            arguments={"sql": sql},
        )
        return LlmResponse(
            content=f'Translating: "{question}"',
            tool_calls=[tool_call],
            finish_reason="tool_calls",
            usage={"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
        )


# ---------------------------------------------------------------------------
# Static user resolver — returns a fixed demo user (no auth needed)
# ---------------------------------------------------------------------------

_DEMO_USER = User(
    id="jtziv",
    username="jontziv",
    email="jontziv@gmail.com",
    group_memberships=[],
)


class StaticUserResolver(UserResolver):
    """Returns a fixed demo user regardless of request context. For demos only."""

    async def resolve_user(self, request_context: RequestContext) -> User:
        return _DEMO_USER


# ---------------------------------------------------------------------------
# Agent factory (used by both CLI demo and sqltranslator server)
# ---------------------------------------------------------------------------

_DB_PATH: Optional[str] = None


def _get_or_create_db() -> str:
    """Return path to the pharma R&D SQLite database, creating it if needed."""
    global _DB_PATH
    if _DB_PATH and os.path.exists(_DB_PATH):
        return _DB_PATH

    # Prefer a stable path in the project root so the file persists between runs
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    db_path = os.path.join(project_root, "pharma_rnd.db")
    build_pharma_db(db_path)
    _DB_PATH = db_path
    return db_path


def create_demo_agent() -> Agent:
    """
    Create the SQLTranslator pharma R&D demo agent (mock LLM, no API key needed).

    Called by:
      - python -m sqltranslator.examples.pharma_rnd_demo
      - python -m sqltranslator.servers --example pharma_rnd_demo
    """
    db_path = _get_or_create_db()

    tool_registry = ToolRegistry()
    sqlite_runner = SqliteRunner(database_path=db_path)
    sql_tool = RunSqlTool(sql_runner=sqlite_runner)
    tool_registry.register_local_tool(sql_tool, access_groups=[])

    llm_service = PharmaRndMockLlmService()

    return Agent(
        llm_service=llm_service,
        tool_registry=tool_registry,
        user_resolver=StaticUserResolver(),
        agent_memory=DemoAgentMemory(),
        config=AgentConfig(
            stream_responses=False,
            include_thinking_indicators=False,
        ),
    )


# ---------------------------------------------------------------------------
# CLI demo runner
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run the SQLTranslator pharma R&D pipeline demo."""
    db_path = _get_or_create_db()

    print("=" * 65)
    print("  SQLTranslator — Pharma R&D Pipeline Intelligence")
    print("  Portfolio Project by Jon Tziv")
    print("=" * 65)
    print(f"  Database : {db_path}")
    print(
        "  Tables   : compounds | clinical_trials | pipeline_history"
        " | milestones | rd_spend"
    )
    print("  Mode     : Mock LLM (no API key required)")
    print("=" * 65)
    print()

    agent = create_demo_agent()
    request_context = RequestContext(metadata={"demo": True})

    for i, (question, _sql) in enumerate(_DEMO_QUERIES, start=1):
        print(f"Query {i}: \"{question}\"")
        print("-" * 65)

        async for component in agent.send_message(
            request_context=request_context,
            message=question,
            conversation_id="pharma-rnd-demo",
        ):
            text = None
            if hasattr(component, "simple_component") and component.simple_component:
                sc = component.simple_component
                if hasattr(sc, "text") and sc.text:
                    text = sc.text
            elif hasattr(component, "rich_component") and component.rich_component:
                rc = component.rich_component
                if hasattr(rc, "content") and rc.content:
                    text = rc.content
            elif hasattr(component, "content") and component.content:
                text = component.content

            if text:
                print(text)

        print()

    print("=" * 65)
    print("  Demo complete — 6 pharma R&D queries executed via natural language")
    print("  Connect a real LLM to enable free-form analyst queries.")
    print("  See README.md for integration options (Claude, GPT, Ollama).")
    print("=" * 65)


def run_interactive() -> None:
    """Entry point for python -m sqltranslator.examples.pharma_rnd_demo"""
    asyncio.run(main())


if __name__ == "__main__":
    run_interactive()
