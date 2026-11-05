"""Stage 6 local Streamlit interface: python -m streamlit run app/app.py."""
import json
import os
from pathlib import Path
import sys
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import pandas as pd
import streamlit as st
from src.app_service import ProjectSession, project_signature
from src.rag import DISCLAIMER, feature_label


def contribution_chart(record: dict[str, Any], cache_dir: Path) -> Any:
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    factors = record["top_factors"]
    labels = [feature_label(f["feature"]).capitalize() for f in factors] + ["Other factors combined"]
    values = [f["shap_value"] * 100 for f in factors] + [record["other_contributions_sum"] * 100]

    fig, ax = plt.subplots(figsize=(8.8, 4.5), layout="constrained")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.barh(
        labels[::-1],
        values[::-1],
        color=["#bc5261" if v > 0 else "#218292" for v in values[::-1]],
        height=0.55,
    )
    ax.axvline(0, color="#8797a2", linewidth=0.8)
    ax.set_xlabel("Contribution to model output (percentage points)", color="#526675")
    ax.tick_params(axis="both", labelsize=10, colors="#344c5d")
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", alpha=0.12)
    ax.set_axisbelow(True)
    return fig


def render_context(result: dict[str, Any]) -> None:
    status = result["knowledge_status"]

    if status == "placeholder_only":
        st.warning(
            "Placeholder knowledge only. No medical evidence has been supplied."
        )
    elif status == "no_relevant_source":
        st.info("No relevant source passed retrieval.")
    else:
        st.info("Retrieved user-supplied literature summaries.")

    st.text(result["text"].split("\nSources:", 1)[0])
    st.caption(result["disclaimer"])

    with st.expander(f"Retrieved sources ({len(result['sources'])})"):
        for source in result["sources"]:
            st.markdown(f"**[{source['citation']}]**")
            st.text(source["title"])
            st.caption(
                "PLACEHOLDER - NOT MEDICAL EVIDENCE"
                if source["status"] == "placeholder"
                else "User-supplied literature summary"
            )
            st.text(source["text"])


def main() -> None:
    st.set_page_config(
        page_title="Clinical Outcomes Explorer",
        page_icon="🩺",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {background: #f5f7fb;}
        .block-container {
            max-width: 1320px;
            padding-top: 2.4rem;
            padding-bottom: 3rem;
        }
        h1 {
            letter-spacing: -0.05em;
            font-weight: 750;
            color: #14283a;
            margin-bottom: 0.25rem;
        }
        h2, h3 {letter-spacing: -0.025em; color: #18364b;}
        div[data-testid="stMetric"] {
            background: #ffffff;
            padding: 1.15rem 1.35rem;
            border: 1px solid #dbe5ee;
            border-radius: 16px;
            box-shadow: 0 4px 16px rgba(21, 48, 68, .05);
        }
        div[data-testid="stMetricLabel"] {
            color: #587084;
            font-size: .82rem;
            text-transform: uppercase;
            letter-spacing: .07em;
        }
        div[data-testid="stMetricValue"] {
            color: #112f45;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid #dce5eb;
            background: #ffffff;
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 1.5rem;
            border-bottom: 1px solid #dbe5ee;
        }
        [data-testid="stTabs"] button {
            height: 48px;
            font-weight: 650;
            color: #587084;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #0d7182;
            border-bottom-color: #0d7182;
        }
        .eyebrow {
            font-size: .75rem;
            letter-spacing: .14em;
            font-weight: 700;
            color: #0d7182;
            margin-bottom: .55rem;
        }
        .hero-copy {
            font-size: 1.12rem;
            color: #587084;
            max-width: 720px;
            margin-bottom: .45rem;
        }
        .outcome-card {
            background: linear-gradient(135deg, #102f45, #135b69);
            color: white;
            padding: 1.7rem 1.8rem;
            border-radius: 18px;
            margin: .5rem 0 1rem;
        }
        .outcome-card .label {
            font-size: .78rem;
            letter-spacing: .12em;
            text-transform: uppercase;
            font-weight: 700;
            opacity: .72;
        }
        .outcome-card .score {
            font-size: 3rem;
            line-height: 1.05;
            font-weight: 750;
            letter-spacing: -.05em;
            margin: .4rem 0;
        }
        .outcome-card p {
            margin: 0;
            opacity: .82;
            max-width: 650px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="eyebrow">CLINICAL ANALYTICS · MIMIC-III DEMO</div>',
        unsafe_allow_html=True,
    )
    st.title("Clinical Outcomes Explorer")
    st.markdown(
        '<div class="hero-copy">Compare two research-model outcomes for an admission, then inspect the mortality model reasoning and source context.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Retrospective research demo · Not clinically validated or calibrated · Not medical advice"
    )

    root = Path(os.environ.get("HEALTHCARE_PROJECT_ROOT", str(CODE_ROOT))).resolve()

    try:
        signature = project_signature(root)
        if st.session_state.get("project_signature") != signature:
            with st.spinner("Loading model and admission data..."):
                st.session_state.project_service = ProjectSession(root)
                st.session_state.project_signature = signature
        service: ProjectSession = st.session_state.project_service
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        st.error("The project is not ready yet.")
        st.text(str(exc))
        st.info("Place CSV files in data/raw and run the preparation commands.")
        st.stop()

    with st.sidebar:
        st.markdown("### Explore admissions")
        st.caption("Choose one record to inspect both research outcomes.")

        scope_label = st.radio(
            "Browse",
            ["Held-out admissions", "All admissions"],
            key="scope",
        )
        catalog = service.admissions(
            "heldout" if scope_label == "Held-out admissions" else "all"
        )

        patients = sorted(catalog.subject_id.unique().tolist())
        subject_id = st.selectbox("Patient ID", patients)
        admissions = catalog.loc[
            catalog.subject_id.eq(subject_id), "hadm_id"
        ].tolist()
        hadm_id = st.selectbox("Admission ID", admissions)

        st.divider()
        st.caption("Mortality model")
        st.markdown(
            "**" + service.bundle["model_name"].replace("_", " ").title() + "**"
        )
        st.caption("Runs locally · No API key needed")

    record = service.model_record(hadm_id)
    secondary_score = service.prolonged_stay_score(hadm_id)

    st.subheader(f"Admission {record['hadm_id']}")
    st.caption(
        f"Patient {record['subject_id']} · "
        + (
            "Held-out patient"
            if record["scope"] == "heldout"
            else "Training patient"
        )
    )

    first, second, third = st.columns(3)
    first.metric("Mortality-model score", f"{record['probability']:.1%}")
    second.metric("Model reference", f"{record['baseline_probability']:.1%}")
    third.metric(
        "Prolonged-stay score",
        "Not prepared" if secondary_score is None else f"{secondary_score:.1%}",
    )

    st.caption(
        "The prolonged-stay score is an equal-weight ensemble of Logistic Regression, Random Forest and XGBoost."
    )

    mortality_tab, stay_tab = st.tabs(
        ["Mortality model", "Length-of-stay model"]
    )

    with mortality_tab:
        st.markdown("### Why did the mortality model give this score?")

        left, right = st.columns([1.15, 1], gap="large")

        with left:
            st.caption(
                "Rose raises the output · Teal lowers it · Relative to the model reference"
            )
            figure = contribution_chart(record, root / "reports/.matplotlib")
            st.pyplot(figure)

            table = pd.DataFrame(
                [
                    {
                        "Factor": feature_label(f["feature"]).capitalize(),
                        "Value": (
                            f"{f['value']:.4g}"
                            if isinstance(f["value"], (float, int))
                            else str(f["value"])
                        ),
                        "Effect (pp)": round(f["shap_value"] * 100, 3),
                    }
                    for f in record["top_factors"]
                ]
            )
            st.dataframe(table, hide_index=True, width=True)

        with right:
            st.markdown("### Explanation and sources")
            try:
                with st.spinner("Preparing explanation..."):
                    result = service.context(record)
                render_context(result)
            except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
                st.warning("The score is available, but the cited context is not ready.")
                st.text(str(exc))

    with stay_tab:
        st.markdown("### Prolonged hospital-stay model")

        if secondary_score is None:
            st.info(
                "Run this once in the terminal: python -m src.prolonged_stay"
            )
        else:
            st.markdown(
                f"""
                <div class="outcome-card">
                    <div class="label">Soft-voting ensemble</div>
                    <div class="score">{secondary_score:.1%}</div>
                    <p>
                        Estimated output for a hospital stay of seven days or more.
                        The length-of-stay feature itself is excluded from this model.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Ensemble design")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Base model": "Logistic Regression", "Weight": "Equal (1/3)"},
                        {"Base model": "Random Forest", "Weight": "Equal (1/3)"},
                        {"Base model": "XGBoost", "Weight": "Equal (1/3)"},
                    ]
                ),
                hide_index=True,
                width=True,
            )

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()