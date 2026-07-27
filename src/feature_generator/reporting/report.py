"""Self-contained HTML report of a single run: promoted features (rationale,
generated code, stability, lift over the features-off baseline) and the
deterministic feature-selection tool's per-iteration combination rounds.

Read-only over the knowledge base and results store -- generating a report
never touches training, sandbox execution, or any LLM call. Rationale/
description/code text embedded in the report originates from the LLM, so the
Jinja2 environment below MUST keep autoescaping on; never mark any of that
content `|safe`.
"""

from __future__ import annotations

from jinja2 import Environment

from feature_generator.config import RunConfig
from feature_generator.knowledge_base.db import KnowledgeBase
from feature_generator.knowledge_base.results_store import ResultsStore


def build_run_report(run_config: RunConfig, run_id: str) -> str:
    kb = KnowledgeBase(run_config.output.knowledge_base_path)
    results_store = ResultsStore(run_config.output.results_store_path)
    try:
        return _render(kb, results_store, run_id)
    finally:
        kb.close()
        results_store.close()


def _fmt_auc(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def _build_feature_context(kb: KnowledgeBase, results_store: ResultsStore, feature_name: str) -> dict:
    detail = results_store.get_promoted_feature_detail(feature_name)
    hypothesis = kb.get_hypothesis(detail["feature_spec"].hypothesis_id)
    validation = detail["validation_result"]
    dynamic = validation.dynamic if validation else None

    stability_entry = None
    if detail["stability_report"]:
        stability_entry = next(
            (f for f in detail["stability_report"].features if f.feature_name == feature_name), None
        )

    training_metrics = detail["training_metrics"]

    return {
        "feature_name": feature_name,
        "feature_type": hypothesis.feature_type if hypothesis else "other",
        "rationale": hypothesis.rationale if hypothesis else "(hypothesis not found)",
        "description": hypothesis.description if hypothesis else "",
        "module_source": detail["module_source"],
        "output_dtype": detail["output_dtype"],
        "single_feature_auc": _fmt_auc(dynamic.single_feature_auc) if dynamic else "n/a",
        "single_feature_auc_flag": dynamic.single_feature_auc_flag if dynamic else False,
        "csi": f"{stability_entry.csi:.4f}" if stability_entry and stability_entry.csi is not None else "n/a",
        "stability_method": stability_entry.method if stability_entry else "n/a",
        "overall_stability_flag": detail["stability_flag"] or "n/a",
        "model_auc": _fmt_auc(detail["auc_mean"]),
        "train_auc": _fmt_auc(training_metrics.train_auc_mean) if training_metrics else "n/a",
        "holdout_auc": _fmt_auc(training_metrics.holdout_auc) if training_metrics else "n/a",
        "lift_over_baseline": _fmt_auc(detail["lift_over_baseline"]),
        "iteration": detail["iteration"],
        "promoted_at": detail["promoted_at"],
    }


def _build_model_performance_rows(kb: KnowledgeBase, run_id: str) -> list[dict]:
    rows = []
    for m in kb.list_training_metrics(run_id):
        rows.append(
            {
                "iteration": m.iteration,
                "train_auc": _fmt_auc(m.train_auc_mean),
                "cv_auc": _fmt_auc(m.auc_mean),
                "cv_auc_std": _fmt_auc(m.auc_std),
                "holdout_auc": _fmt_auc(m.holdout_auc),
            }
        )
    return rows


def _build_round_context(round_dict: dict) -> dict:
    baseline_auc = round_dict.get("baseline_auc")
    previous_auc = baseline_auc
    steps = []
    for name, auc in round_dict.get("auc_trace", []):
        lift = auc - previous_auc if previous_auc is not None else None
        steps.append({"feature_name": name, "auc": _fmt_auc(auc), "lift": _fmt_auc(lift)})
        previous_auc = auc

    return {
        "iteration": round_dict["iteration"],
        "baseline_auc": _fmt_auc(baseline_auc),
        "final_auc": _fmt_auc(round_dict.get("final_auc")),
        "accepted_candidates": round_dict.get("accepted_candidates", []),
        "rejected_by_lift": round_dict.get("rejected_by_lift", []),
        "redundancy_clusters": [c for c in round_dict.get("redundancy_clusters", []) if len(c) > 1],
        "final_selected": round_dict.get("final_selected", []),
        "steps": steps,
    }


def _build_hypotheses_rows(kb: KnowledgeBase, run_id: str) -> list[dict]:
    hypotheses = kb.list_hypotheses(run_id)
    results_by_hypothesis = {r.hypothesis_id: r for r in kb.list_validation_results(run_id)}
    rows = []
    for h in hypotheses:
        result = results_by_hypothesis.get(h.id)
        rows.append(
            {
                "iteration": h.iteration,
                "description": h.description,
                "feature_type": h.feature_type,
                "status": result.final_status if result else "pending",
            }
        )
    return rows


def _render(kb: KnowledgeBase, results_store: ResultsStore, run_id: str) -> str:
    run_metadata = kb.get_run_metadata(run_id)
    best_metrics = kb.get_best_training_metrics(run_id)
    promoted_summaries = results_store.list_promoted_features(run_id)

    context = {
        "run_id": run_id,
        "dataset_name": run_metadata["dataset_name"] if run_metadata else "unknown",
        "baseline_auc": _fmt_auc(run_metadata["baseline_auc"] if run_metadata else None),
        "raw_feature_columns": run_metadata["raw_feature_columns"] if run_metadata else [],
        "best_auc": _fmt_auc(best_metrics.auc_mean if best_metrics else None),
        "promoted_features": [
            _build_feature_context(kb, results_store, row["feature_name"]) for row in promoted_summaries
        ],
        "selection_rounds": [
            _build_round_context(r) for r in kb.list_feature_selection_rounds(run_id)
        ],
        "model_performance": _build_model_performance_rows(kb, run_id),
        "hypotheses": _build_hypotheses_rows(kb, run_id),
    }

    env = Environment(autoescape=True)
    return env.from_string(_TEMPLATE).render(**context)


_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Run report -- {{ run_id }}</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #fafafa; --fg: #14161a; --muted: #4b4f58;
    --card-bg: #ffffff; --border: #d5d7db;
    --code-bg: #eef0f3; --code-fg: #1a1a1a;
    --table-header-bg: #e8eaee;
    --badge-bg: #dbe4ff; --badge-fg: #1f2f8c;
    --warn-bg: #ffe3cc; --warn-fg: #7a3600;
    --ok-fg: #0f6b2c; --bad-fg: #9a1a10;
    --link: #1a4fd6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14161a; --fg: #f2f3f5; --muted: #c7cbd3;
      --card-bg: #1f232b; --border: #454a55;
      --code-bg: #10131a; --code-fg: #e8eaef;
      --table-header-bg: #2a2f3a;
      --badge-bg: #2c3868; --badge-fg: #c4d0ff;
      --warn-bg: #4a2c10; --warn-fg: #ffc38a;
      --ok-fg: #5fd685; --bad-fg: #ff8a7a;
      --link: #8fb4ff;
    }
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 980px; margin: 2rem auto; padding: 0 1.5rem;
    background: var(--bg); color: var(--fg); line-height: 1.5;
  }
  a { color: var(--link); }
  h1 { margin-bottom: 0.2rem; }
  .subtitle { color: var(--muted); font-weight: 600; margin-top: 0; }
  .summary { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }
  .stat { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 0.8rem 1.2rem; min-width: 160px; }
  .stat .label { font-size: 0.78rem; color: var(--muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }
  .stat .value { font-size: 1.5rem; font-weight: 700; }
  h2 { border-bottom: 2px solid var(--border); padding-bottom: 0.3rem; margin-top: 2.5rem; }
  .card {
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px;
    padding: 1.2rem 1.4rem; margin-bottom: 1.2rem;
  }
  .card h3 { margin: 0 0 0.3rem 0; }
  .badge {
    display: inline-block; font-size: 0.75rem; padding: 0.1rem 0.5rem; border-radius: 6px;
    background: var(--badge-bg); color: var(--badge-fg); margin-left: 0.4rem; font-weight: 700;
  }
  .badge.warn { background: var(--warn-bg); color: var(--warn-fg); }
  .metrics-row { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 0.8rem 0; font-size: 0.92rem; }
  .metrics-row div { background: var(--table-header-bg); border-radius: 6px; padding: 0.3rem 0.7rem; }
  .rationale { color: var(--fg); margin: 0.6rem 0; }
  .rationale strong { font-weight: 700; }
  details > summary { cursor: pointer; font-weight: 700; margin-top: 0.6rem; }
  pre { background: var(--code-bg); color: var(--code-fg); border-radius: 8px; padding: 0.9rem; overflow-x: auto; font-size: 0.85rem; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  table { border-collapse: collapse; width: 100%; background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  th, td { text-align: left; padding: 0.5rem 0.7rem; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
  th { background: var(--table-header-bg); font-weight: 700; }
  .status-validated, .status-promoted { color: var(--ok-fg); font-weight: 700; }
  .status-static_failed, .status-dynamic_failed, .status-leakage_rejected, .status-serving_mismatch { color: var(--bad-fg); font-weight: 700; }
  .empty { color: var(--muted); font-weight: 600; font-style: italic; }
</style>
</head>
<body>

<h1>Run report</h1>
<p class="subtitle">{{ dataset_name }} &middot; run <code>{{ run_id }}</code></p>

<div class="summary">
  <div class="stat"><div class="label">Features-off baseline AUC</div><div class="value">{{ baseline_auc }}</div></div>
  <div class="stat"><div class="label">Best achieved AUC</div><div class="value">{{ best_auc }}</div></div>
  <div class="stat"><div class="label">Raw feature columns</div><div class="value">{{ raw_feature_columns|length }}</div></div>
  <div class="stat"><div class="label">Promoted features</div><div class="value">{{ promoted_features|length }}</div></div>
</div>

<h2>Model performance over the run</h2>
<p class="rationale">Train AUC comes from each fold's own training rows (overfit-gap signal).
CV AUC is the mean &plusmn; std across validation folds -- what drives feature-selection
decisions. Holdout AUC is a model fit on dev rows only, scored once against rows the CV loop
and the LLM never see -- the honest check against overfitting to the CV procedure itself.</p>
{% if not model_performance %}
<p class="empty">No training runs recorded yet.</p>
{% else %}
<table>
  <tr><th>Iteration</th><th>Train AUC</th><th>CV AUC (mean &plusmn; std)</th><th>Holdout AUC</th></tr>
  {% for m in model_performance %}
  <tr>
    <td>{{ m.iteration }}</td>
    <td>{{ m.train_auc }}</td>
    <td>{{ m.cv_auc }} &plusmn; {{ m.cv_auc_std }}</td>
    <td>{{ m.holdout_auc }}</td>
  </tr>
  {% endfor %}
</table>
{% endif %}

<h2>Promoted features</h2>
{% if not promoted_features %}
<p class="empty">No features have been promoted to the results store yet.</p>
{% endif %}
{% for f in promoted_features %}
<div class="card">
  <h3>{{ f.feature_name }}
    <span class="badge">{{ f.feature_type }}</span>
    <span class="badge">{{ f.output_dtype }}</span>
    {% if f.single_feature_auc_flag %}<span class="badge warn">high single-feature AUC</span>{% endif %}
  </h3>
  <p class="rationale"><strong>Rationale:</strong> {{ f.rationale }}</p>
  <p class="rationale"><strong>Computed as:</strong> {{ f.description }}</p>
  <div class="metrics-row">
    <div>Lift over baseline: <strong>{{ f.lift_over_baseline }}</strong></div>
    <div>Train AUC: <strong>{{ f.train_auc }}</strong></div>
    <div>CV AUC at promotion: <strong>{{ f.model_auc }}</strong></div>
    <div>Holdout AUC: <strong>{{ f.holdout_auc }}</strong></div>
    <div>Single-feature AUC: <strong>{{ f.single_feature_auc }}</strong></div>
    <div>Stability (CSI, {{ f.stability_method }}): <strong>{{ f.csi }}</strong> ({{ f.overall_stability_flag }})</div>
    <div>Found in iteration {{ f.iteration }}</div>
  </div>
  <details>
    <summary>Generated code</summary>
    <pre><code>{{ f.module_source }}</code></pre>
  </details>
</div>
{% endfor %}

<h2>Feature-selection rounds</h2>
{% if not selection_rounds %}
<p class="empty">No candidate features reached the selection tool yet.</p>
{% endif %}
{% for r in selection_rounds %}
<div class="card">
  <h3>Iteration {{ r.iteration }}: {{ r.baseline_auc }} &rarr; {{ r.final_auc }} AUC</h3>
  <p><strong>Accepted by single-feature lift screen:</strong>
    {{ r.accepted_candidates|join(', ') if r.accepted_candidates else '(none)' }}</p>
  <p><strong>Rejected by single-feature lift screen:</strong>
    {{ r.rejected_by_lift|join(', ') if r.rejected_by_lift else '(none)' }}</p>
  {% if r.redundancy_clusters %}
  <p><strong>Redundancy clusters (correlated candidates merged):</strong></p>
  <ul>
    {% for cluster in r.redundancy_clusters %}<li>{{ cluster|join(' ~ ') }}</li>{% endfor %}
  </ul>
  {% endif %}
  {% if r.steps %}
  <table>
    <tr><th>Greedy search step</th><th>Resulting AUC</th><th>Step lift</th></tr>
    {% for step in r.steps %}
    <tr><td>{{ step.feature_name }}</td><td>{{ step.auc }}</td><td>{{ step.lift }}</td></tr>
    {% endfor %}
  </table>
  {% endif %}
  <p><strong>Final selected set:</strong> {{ r.final_selected|join(', ') if r.final_selected else '(baseline only)' }}</p>
</div>
{% endfor %}

<h2>All hypotheses this run</h2>
{% if not hypotheses %}
<p class="empty">No hypotheses recorded yet.</p>
{% else %}
<table>
  <tr><th>Iter</th><th>Description</th><th>Type</th><th>Status</th></tr>
  {% for h in hypotheses %}
  <tr>
    <td>{{ h.iteration }}</td>
    <td>{{ h.description }}</td>
    <td>{{ h.feature_type }}</td>
    <td class="status-{{ h.status }}">{{ h.status }}</td>
  </tr>
  {% endfor %}
</table>
{% endif %}

</body>
</html>
"""
