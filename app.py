"""
SQLTranslator — Pharma R&D Data Intelligence
Streamlit MVP · Powered by Groq + Open-Source LLMs
"""

import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Load .env before anything reads os.environ ───────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

# ── Page config (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="SQLTranslator · Pharma R&D",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH      = Path(__file__).parent / "pharma_rnd.db"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Three current Groq open-source models (verified Feb 2026)
MODELS = {
    "⚡ Llama 3.3 · 70B  — Best quality":  "llama-3.3-70b-versatile",
    "🦙 Llama 4 Scout · 17B — Latest":     "meta-llama/llama-4-scout-17b-16e-instruct",
    "🚀 Llama 3.1 · 8B  — Fastest":        "llama-3.1-8b-instant",
}

SCHEMA_CONTEXT = """
Database: pharma_rnd.db (SQLite)

Tables and columns:
  compounds         — compound_id PK, name, therapeutic_area, mechanism, indication,
                      current_phase, status, first_in_class BOOL,
                      projected_peak_sales_m FLOAT, entry_date DATE
  clinical_trials   — trial_id PK, compound_id FK, phase, status, indication,
                      sites INT, target_enrollment INT, actual_enrollment INT,
                      start_date DATE, primary_completion DATE, serious_adverse_events INT
  pipeline_history  — event_id PK, compound_id FK, from_phase, to_phase,
                      event_date DATE, outcome
  milestones        — milestone_id PK, compound_id FK, milestone_type,
                      planned_date DATE, actual_date DATE, status, days_variance INT
  rd_spend          — spend_id PK, therapeutic_area, fiscal_year INT,
                      phase, spend_usd_m FLOAT

Allowed values:
  therapeutic_area  : Oncology, Virology, Immunology, Neurology, Cardiology
  current_phase     : Phase I, Phase II, Phase III, Approved
  status(compounds) : Active, Discontinued, Filed
  status(trials)    : Active, Completed, Terminated
  outcome           : Advanced, Failed, Paused
  milestone_type    : IND Filing, Phase Start, NDA Submission, FDA Decision
"""

SQL_SYSTEM_PROMPT = f"""You are an expert SQLite analyst for a pharmaceutical R&D database.
Your ONLY output must be a single raw SQLite SELECT query — nothing else.
No markdown, no backticks, no triple quotes, no explanation, no comments, no preamble.

{SCHEMA_CONTEXT}

Hard rules:
- Output the bare SQL and absolutely nothing else
- Use only SQLite-compatible syntax (no ILIKE, no FETCH NEXT, no QUALIFY, no PIVOT)
- JOIN tables via compound_id
- Cap results at 200 rows with LIMIT unless the user requests all
- Never emit DROP, DELETE, INSERT, UPDATE, CREATE, ALTER, TRUNCATE
"""

SAMPLE_QUESTIONS = [
    "How many compounds are in each clinical phase?",
    "Which therapeutic areas have the most pipeline compounds?",
    "Show active Phase III trials with enrollment below target",
    "Average projected peak sales by therapeutic area",
    "Which compounds have serious adverse events in their trials?",
    "Compare R&D spend across therapeutic areas for 2023",
    "Show all first-in-class compounds and their current phase",
    "Which compounds missed milestones by more than 60 days?",
    "List compounds whose trials are behind enrollment target",
    "Attrition rate at each clinical phase",
]

STARTER_CARDS = [
    ("🏥", "Phase Breakdown",
     "How many compounds are in each clinical phase?"),
    ("📈", "Top Performers",
     "Show all first-in-class compounds and their current phase"),
    ("💰", "R&D Spend",
     "Compare R&D spend across therapeutic areas for 2023"),
]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Layout ── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #252d3d; }
[data-testid="stMain"]             { padding-top: 1rem; }

/* ── Typography ── */
h1, h2, h3, h4 { color: #e2e8f0 !important; }
p, li, label   { color: #94a3b8 !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a2235 0%, #1e1b4b 100%);
    border: 1px solid #252d3d; border-radius: 16px;
    padding: 1.4rem 2rem; margin-bottom: 1.4rem;
}

/* ── Chat bubbles ── */
.user-bubble {
    background: #1e2d4a; border: 1px solid #2d4a7a;
    border-radius: 12px 12px 4px 12px;
    padding: .75rem 1.1rem; margin: .6rem 0;
    color: #93c5fd; font-size: .95rem; line-height: 1.5;
}
.assistant-wrap {
    background: #1a2235; border: 1px solid #252d3d;
    border-radius: 12px 12px 12px 4px;
    padding: .85rem 1.1rem; margin: .6rem 0;
}

/* ── SQL code block ── */
.sql-block {
    background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
    padding: .8rem 1rem;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: .82rem; color: #7dd3fc;
    white-space: pre-wrap; word-break: break-word;
    line-height: 1.55;
}

/* ── Badges ── */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: .72rem; font-weight: 600; letter-spacing: .04em; margin-right: 4px;
}
.bg { background:#052e16; color:#4ade80; border:1px solid #166534; }
.bb { background:#0c1a3a; color:#60a5fa; border:1px solid #1d4ed8; }
.bp { background:#1e1b4b; color:#a78bfa; border:1px solid #4338ca; }
.ba { background:#1c1007; color:#fbbf24; border:1px solid #92400e; }

/* ── Status dots ── */
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px; vertical-align:middle; }
.ok  { background:#4ade80; box-shadow:0 0 5px #4ade80; }
.err { background:#f87171; box-shadow:0 0 5px #f87171; }

/* ── Quick-start card ── */
.qs-card {
    background: #1a2235; border: 1px solid #252d3d;
    border-left: 3px solid #6366f1; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: .5rem; text-align: center;
}

/* ── Input ── */
[data-testid="stTextInput"] input {
    background: #1a2235 !important; border: 1px solid #252d3d !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
    font-size: .95rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,.2) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #475569 !important; }

/* ── Primary button ── */
button[kind="primary"], [data-testid="stButton"] > button {
    background: #6366f1 !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; transition: background .15s !important;
}
button[kind="primary"]:hover, [data-testid="stButton"] > button:hover {
    background: #4f46e5 !important;
}

/* ── Sidebar sample buttons (override to look subtle) ── */
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background: #1a2235 !important; color: #94a3b8 !important;
    border: 1px solid #252d3d !important; font-weight: 400 !important;
    font-size: .81rem !important; text-align: left !important;
    transition: border-color .15s, color .15s !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    border-color: #6366f1 !important; color: #a5b4fc !important;
    background: #1e2748 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #252d3d !important; border-radius: 8px !important;
    background: #0f1117 !important;
}
[data-testid="stExpander"] summary { color: #64748b !important; font-size: .85rem !important; }
[data-testid="stExpander"] summary:hover { color: #94a3b8 !important; }

/* ── Metrics ── */
[data-testid="stMetricValue"] { color: #6366f1 !important; font-size: 1.45rem !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: .78rem !important; }
[data-testid="stMetricDelta"] { font-size: .75rem !important; }

/* ── Selectbox ── */
[data-baseweb="select"] > div { background: #1a2235 !important; border-color: #252d3d !important; }
[data-baseweb="select"] span  { color: #e2e8f0 !important; }

/* ── Divider ── */
hr { border-color: #252d3d !important; margin: .9rem 0 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] iframe { border-radius: 8px; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: #1a2235 !important; color: #94a3b8 !important;
    border: 1px solid #252d3d !important; font-size: .82rem !important;
    border-radius: 6px !important; padding: .3rem .9rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: #6366f1 !important; color: #a5b4fc !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #6366f1 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "auto_submit"  not in st.session_state: st.session_state.auto_submit  = None


# ── DB helpers ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_db_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    stats = {}
    for t in ["compounds", "clinical_trials", "pipeline_history", "milestones", "rd_spend"]:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        stats[t] = cur.fetchone()[0]
    conn.close()
    return stats


def run_sql(query: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df


def clean_sql(raw: str) -> str:
    """Strip markdown fences and leading prose from LLM output."""
    raw = raw.strip()
    # Remove markdown fences
    raw = re.sub(r"^```sql\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*",    "", raw)
    raw = re.sub(r"```\s*$",    "", raw)
    # Remove Qwen/DeepSeek <think>…</think> blocks
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    # Discard any leading lines that aren't SQL keywords
    lines = raw.strip().splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"^\s*(SELECT|WITH|--)", ln, re.IGNORECASE):
            return "\n".join(lines[i:]).strip()
    return raw.strip()


# ── Groq SQL generation ───────────────────────────────────────────────────────
def generate_sql(question: str, model_id: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not configured. "
            "Add it to a .env file in the project root or export it as an environment variable."
        )
    from groq import Groq
    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    return clean_sql(response.choices[0].message.content)


# ── Visualisation ─────────────────────────────────────────────────────────────
THEME = dict(
    paper_bgcolor="#1a2235",
    plot_bgcolor="#0f1117",
    font_color="#94a3b8",
    margin=dict(t=30, b=20, l=10, r=10),
)
COLORS = px.colors.qualitative.Vivid


def build_chart(df: pd.DataFrame) -> go.Figure | None:
    """Return the most appropriate Plotly figure for the dataframe, or None."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    str_cols = df.select_dtypes(include="object").columns.tolist()

    if not num_cols:
        return None

    y_col = num_cols[0]

    if str_cols:
        x_col = str_cols[0]
        n     = len(df)

        # Pie for small category counts with known group columns
        pie_keywords = ("area", "phase", "status", "outcome", "type", "indication")
        if n <= 10 and any(k in x_col.lower() for k in pie_keywords):
            fig = px.pie(
                df, names=x_col, values=y_col, hole=0.4,
                color_discrete_sequence=COLORS,
            )
            fig.update_traces(
                textposition="inside", textinfo="percent+label",
                textfont_color="#e2e8f0",
                hovertemplate="<b>%{label}</b><br>%{value}<extra></extra>",
            )
            fig.update_layout(**THEME, showlegend=True, legend_font_color="#94a3b8")
            return fig

        # Horizontal bar for everything else
        plot_df = df.head(20).copy()
        fig = px.bar(
            plot_df, x=y_col, y=x_col, orientation="h",
            color=y_col, color_continuous_scale="Viridis",
            text=y_col,
        )
        fig.update_traces(
            texttemplate="%{text:.3s}",
            textposition="outside",
            textfont_color="#e2e8f0",
            hovertemplate="<b>%{y}</b>: %{x}<extra></extra>",
        )
        fig.update_layout(
            **THEME,
            xaxis=dict(gridcolor="#252d3d", color="#94a3b8", title=None),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", color="#94a3b8", title=None),
            coloraxis_showscale=False,
            height=max(300, len(plot_df) * 42),
            margin=dict(t=20, b=20, l=200, r=80),
        )
        return fig

    # Multiple numeric columns → grouped bar
    if len(num_cols) >= 2:
        fig = px.bar(
            df.head(50), barmode="group",
            x=df.head(50).index,
            y=num_cols[:4],
            color_discrete_sequence=COLORS,
        )
        fig.update_layout(
            **THEME,
            xaxis=dict(gridcolor="#252d3d", color="#94a3b8"),
            yaxis=dict(gridcolor="#252d3d", color="#94a3b8"),
            legend_font_color="#94a3b8",
        )
        return fig

    return None


def render_chart(df: pd.DataFrame, key: str) -> None:
    """Render the chart with a key, or display an informational message."""
    try:
        fig = build_chart(df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{key}")
        else:
            st.caption("No numeric columns to visualise for this result.")
    except Exception as exc:
        st.caption(f"Chart could not be rendered: {exc}")


# ── Core processing ───────────────────────────────────────────────────────────
def process_question(question: str, model_id: str, model_label: str) -> None:
    """Generate SQL, run it, store result in session_state, then rerun."""
    question = question.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})

    payload: dict = {
        "question":    question,
        "model_label": model_label,
        "sql":         None,
        "df":          None,
        "error":       None,
    }

    with st.spinner(f"🤖 Writing SQL with {model_label.split('·')[0].strip()}…"):
        try:
            payload["sql"] = generate_sql(question, model_id)
        except Exception as exc:
            payload["error"] = str(exc)
            st.session_state.messages.append({"role": "assistant", "content": payload})
            st.rerun()
            return

    with st.spinner("⚡ Running query…"):
        try:
            payload["df"] = run_sql(payload["sql"])
        except Exception as exc:
            payload["error"] = (
                f"SQL execution failed: {exc}\n\n"
                f"Generated SQL:\n```sql\n{payload['sql']}\n```"
            )

    st.session_state.messages.append({"role": "assistant", "content": payload})
    st.rerun()


# ── Render one assistant turn ─────────────────────────────────────────────────
def render_assistant(payload: dict, idx: int) -> None:
    """Render SQL, data table, chart, and download for a single assistant message."""

    # ── Error state ──
    if payload.get("error"):
        st.error(payload["error"])
        if payload.get("sql"):
            with st.expander("📄 Generated SQL"):
                st.markdown(
                    f'<div class="sql-block">{payload["sql"]}</div>',
                    unsafe_allow_html=True,
                )
        return

    # ── SQL expander ──
    with st.expander("📄 View SQL", expanded=False):
        st.markdown(
            f'<div class="sql-block">{payload["sql"]}</div>',
            unsafe_allow_html=True,
        )

    df: pd.DataFrame | None = payload.get("df")
    if df is None or df.empty:
        st.info("The query returned no rows.")
        return

    rows = len(df)
    cols = len(df.columns)

    # ── Row / column badges ──
    ml = payload.get("model_label", "")
    # Strip emoji prefix and trailing whitespace for the badge text
    ml_short = re.sub(r"^[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf⚡🦙🚀]\s*", "", ml).strip()
    st.markdown(
        f'<span class="badge bg">{rows} rows</span>'
        f'<span class="badge bb">{cols} cols</span>'
        f'<span class="badge bp">{ml_short}</span>',
        unsafe_allow_html=True,
    )

    # ── Data table ──
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=min(420, max(100, (rows + 1) * 35 + 42)),
    )

    # ── Visualisation ──
    if rows >= 2:
        with st.expander("📊 Visualisation", expanded=True):
            render_chart(df, key=str(idx))

    # ── Download ──
    st.download_button(
        label="⬇️ Download CSV",
        data=df.to_csv(index=False).encode(),
        file_name="query_results.csv",
        mime="text/csv",
        key=f"dl_{idx}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:.8rem 0 .4rem;">
        <span style="font-size:2.2rem;">🧬</span>
        <h2 style="margin:.2rem 0 0;color:#e2e8f0;font-size:1.1rem;font-weight:700;">
            SQLTranslator
        </h2>
        <p style="margin:0;font-size:.75rem;color:#475569;">
            Pharma R&amp;D Intelligence
        </p>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ── Model picker (resolved before sample buttons so it's available in callbacks) ──
    st.markdown("#### 🤖 AI Model")
    model_label = st.selectbox(
        "model",
        list(MODELS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    selected_model_id = MODELS[model_label]

    # API status pill
    if GROQ_API_KEY:
        st.markdown(
            '<span class="dot ok"></span>'
            '<span style="font-size:.8rem;color:#4ade80;">Groq API · ready</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="dot err"></span>'
            '<span style="font-size:.8rem;color:#f87171;">No GROQ_API_KEY found</span>',
            unsafe_allow_html=True,
        )
        st.caption("Create a `.env` file with `GROQ_API_KEY=gsk_...` and restart.")

    st.markdown("---")

    # ── Database stats ──
    st.markdown("#### 🗄️ Database")
    try:
        stats = get_db_stats()
        c1, c2 = st.columns(2)
        c1.metric("Compounds",  stats["compounds"])
        c2.metric("Trials",     stats["clinical_trials"])
        c1.metric("Milestones", stats["milestones"])
        c2.metric("Spend rows", stats["rd_spend"])
        st.markdown(
            '<span class="dot ok"></span>'
            '<span style="font-size:.8rem;color:#4ade80;">pharma_rnd.db</span>',
            unsafe_allow_html=True,
        )
    except Exception as db_err:
        st.error(f"DB error: {db_err}")

    st.markdown("---")

    # ── Sample questions — click → instant submit ──
    st.markdown("#### 💡 Sample Questions")
    st.caption("Click any question to run it instantly")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sq_{q[:30]}", use_container_width=True):
            st.session_state.auto_submit = q

    st.markdown("---")

    if st.button("🗑️ Clear conversation", use_container_width=True, key="clear_btn"):
        st.session_state.messages    = []
        st.session_state.auto_submit = None
        st.rerun()

    st.markdown(
        '<p style="font-size:.7rem;color:#374151;text-align:center;margin-top:.6rem;">'
        "Built by Jon Tziv · Powered by Groq</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

# ── Auto-submit from sidebar sample click ─────────────────────────────────────
if st.session_state.auto_submit:
    pending = st.session_state.auto_submit
    st.session_state.auto_submit = None          # clear before processing
    process_question(pending, selected_model_id, model_label)
    # process_question ends with st.rerun() so code below won't execute this run

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1 style="margin:0;font-size:1.55rem;color:#e2e8f0;">
        🧬 SQLTranslator — Pharma R&amp;D Intelligence
    </h1>
    <p style="margin:.35rem 0 0;color:#94a3b8;font-size:.93rem;">
        Ask any question in plain English.
        Get SQL, data, and charts instantly — no SQL knowledge required.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
assistant_idx = 0
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">👤 &nbsp;{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="assistant-wrap">', unsafe_allow_html=True)
        render_assistant(msg["content"], idx=assistant_idx)
        st.markdown("</div>", unsafe_allow_html=True)
        assistant_idx += 1

# ── Input bar ─────────────────────────────────────────────────────────────────
st.markdown("---")

col_q, col_btn = st.columns([5, 1])
with col_q:
    user_question = st.text_input(
        "question",
        placeholder='Ask anything, e.g. "Which Phase III compounds have the highest sales projections?"',
        label_visibility="collapsed",
        key="user_input",
    )
with col_btn:
    submit = st.button("Ask →", use_container_width=True, key="ask_btn")

st.caption("Type a question or click a sample from the sidebar — the AI writes and runs the SQL for you.")

if submit and user_question.strip():
    process_question(user_question, selected_model_id, model_label)

# ── Empty state / Quick-start cards ──────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div style="text-align:center;padding:2rem 1rem 1.2rem;">
    <div style="font-size:3rem;margin-bottom:.7rem;">🔬</div>
    <h3 style="color:#475569;font-weight:500;margin-bottom:.35rem;">No questions yet</h3>
    <p style="color:#374151;font-size:.88rem;max-width:460px;margin:0 auto;">
        Type a question above or try one of the quick-starts below.
        Click any sample in the sidebar to run it instantly.
    </p>
</div>""", unsafe_allow_html=True)

    st.markdown("#### Quick-start →")
    qcols = st.columns(3)
    for col, (icon, title, q) in zip(qcols, STARTER_CARDS):
        with col:
            st.markdown(f"""
<div class="qs-card">
    <div style="font-size:1.6rem;">{icon}</div>
    <p style="color:#e2e8f0;font-weight:600;margin:.3rem 0 .2rem;font-size:.92rem;">{title}</p>
    <p style="font-size:.78rem;color:#64748b;margin:0;">"{q}"</p>
</div>""", unsafe_allow_html=True)
            if st.button("Try it", key=f"qs_{title}", use_container_width=True):
                process_question(q, selected_model_id, model_label)
