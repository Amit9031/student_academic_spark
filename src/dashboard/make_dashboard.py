"""Build a simple visual dashboard (one HTML file, no internet or extra packages needed).

Numbers are computed with Spark from the CSV; model results are read from
output/csv/model_metrics.json (created by src/ml/simple_model.py).

Run:  python src/dashboard/make_dashboard.py
Then: open output/dashboard.html
"""
import html
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spark_common import get_spark, read_students, OUTPUT_DIR  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

CSS = """
:root{--bg:#f5f6fa;--card:#fff;--text:#1f2937;--muted:#6b7280;--accent:#2563eb;--bar:#3b82f6;--bad:#dc2626;--ok:#16a34a;--line:#e5e7eb}
@media (prefers-color-scheme:dark){:root{--bg:#0f172a;--card:#1e293b;--text:#e5e7eb;--muted:#94a3b8;--accent:#60a5fa;--bar:#60a5fa;--line:#334155}}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text)}
header{padding:24px 32px}h1{margin:0;font-size:26px}.sub{color:var(--muted);margin-top:4px;font-size:14px}
main{padding:0 32px 40px;max-width:1200px;margin:auto}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:18px}
.card{overflow-x:auto}.kpi,.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.kpi .v{font-size:30px;font-weight:700;color:var(--accent)}.kpi .l{color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
h2{font-size:16px;margin:0 0 12px}table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:6px 6px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600}
.note{color:var(--muted);font-size:12px;margin-top:10px;line-height:1.5}.bad{color:var(--bad);font-weight:600}
svg text{fill:var(--text);font-size:12px}svg .muted{fill:var(--muted)}
"""


def bar_chart(items, width=420, bar_h=24, color="var(--bar)", fmt="{:.2f}"):
    """Horizontal bar chart as inline SVG. items = [(label, value), ...]"""
    if not items:
        return "<p class='note'>No data</p>"
    vmax = max(v for _, v in items) or 1
    left, gap = 110, 8
    h = len(items) * (bar_h + gap) + gap
    rows = []
    for i, (label, v) in enumerate(items):
        y = gap + i * (bar_h + gap)
        w = max(2, (width - left - 60) * v / vmax)
        rows.append(f'<text x="{left - 8}" y="{y + bar_h * 0.7}" text-anchor="end">{html.escape(str(label))}</text>'
                    f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" style="fill:{color}"/>'
                    f'<text x="{left + w + 6:.1f}" y="{y + bar_h * 0.7}">{fmt.format(v)}</text>')
    return f'<svg viewBox="0 0 {width} {h}" width="100%" role="img">{"".join(rows)}</svg>'


def column_chart(items, width=420, height=190):
    """Vertical bars for a histogram. items = [(label, count), ...]"""
    vmax = max(v for _, v in items) or 1
    n = len(items)
    pad_l, pad_b, pad_t = 8, 22, 16
    bw = (width - pad_l * 2) / n
    out = []
    for i, (label, v) in enumerate(items):
        bh = (height - pad_b - pad_t) * v / vmax
        x = pad_l + i * bw
        y = height - pad_b - bh
        out.append(f'<rect x="{x + 2:.1f}" y="{y:.1f}" width="{bw - 4:.1f}" height="{bh:.1f}" rx="3" style="fill:var(--bar)"/>'
                   f'<text x="{x + bw / 2:.1f}" y="{height - 6}" text-anchor="middle" class="muted">{label}</text>'
                   f'<text x="{x + bw / 2:.1f}" y="{y - 3:.1f}" text-anchor="middle">{v if v else ""}</text>')
    return f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">{"".join(out)}</svg>'


def table(headers, rows):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def model_section(metrics):
    if not metrics:
        return ("<div class='card'><h2>Model results</h2><p class='note'>No model results yet. "
                "Run <code>python src/ml/simple_model.py</code> and rebuild the dashboard.</p></div>")
    lin_rows = [[n.replace("linear_", ""), f"{m['r2']:.3f}", f"{m['rmse']:.2f}", f"{m['mae']:.2f}",
                 f"{m['baseline_rmse']:.2f}"] for n, m in metrics.items() if n.startswith("linear_")]
    log_rows = [[n.replace("logistic_", ""), f"{m['accuracy']:.3f}", f"{m['precision']:.3f}",
                 f"{m['recall']:.3f}", f"{m['f1']:.3f}", f"{m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}"]
                for n, m in metrics.items() if n.startswith("logistic_")]
    return (
        "<div class='card'><h2>Linear regression: predict G3</h2>"
        + table(["Scenario", "R\u00b2", "RMSE", "MAE", "Baseline RMSE"], lin_rows)
        + "<p class='note'>EARLY = background features only. GRADES = also G1 and G2. "
          "Baseline RMSE is the error of always guessing the average mark.</p></div>"
        "<div class='card'><h2>Logistic regression: predict FAIL (G3 &lt; 10)</h2>"
        + table(["Scenario", "Accuracy", "Precision", "Recall", "F1", "TP/FP/TN/FN"], log_rows)
        + "<p class='note'>Positive class = FAIL. Recall = share of failing students the model caught.</p></div>")


def main():
    spark = get_spark("Dashboard")
    df = read_students(spark).withColumn(
        "category", F.when(F.col("G3") >= 15, "Excellent").when(F.col("G3") >= 10, "Pass").otherwise("At-Risk")).cache()

    total = df.count()
    avg_g3 = df.agg(F.avg("G3")).first()[0]
    pass_n = df.filter(F.col("G3") >= 10).count()
    risk_n = total - pass_n
    avg_abs = df.agg(F.avg("absences")).first()[0]

    def avg_by(col):
        return [(r[col], r["a"]) for r in df.groupBy(col).agg(F.round(F.avg("G3"), 2).alias("a")).orderBy(col).collect()]

    by_school = [(f"School {k}", v) for k, v in avg_by("school")]
    by_sex = [({"F": "Female", "M": "Male"}.get(k, k), v) for k, v in avg_by("sex")]
    by_study = [(f"Study time {k}", v) for k, v in avg_by("studytime")]
    by_fail = [(f"{k} past failures", v) for k, v in avg_by("failures")]
    cats = dict(df.groupBy("category").count().rdd.map(lambda r: (r[0], r[1])).collect())
    cat_items = [(c, cats.get(c, 0)) for c in ("Excellent", "Pass", "At-Risk")]
    hist = dict(df.groupBy("G3").count().rdd.map(lambda r: (r[0], r[1])).collect())
    hist_items = [(g, hist.get(g, 0)) for g in range(0, 21)]
    risk_rows = [[r["school"], r["sex"], r["age"], r["studytime"], r["failures"], r["absences"], r["G1"], r["G2"],
                  f"<span class='bad'>{r['G3']}</span>"]
                 for r in df.filter(F.col("G3") < 10).orderBy("G3", F.desc("absences")).limit(10).collect()]

    metrics_path = OUTPUT_DIR / "csv" / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Student Performance Dashboard</title>
<style>{CSS}</style></head><body>
<header><h1>Student Performance Dashboard</h1>
<div class="sub">Mathematics course (student-mat.csv) &middot; processed with Apache Spark &middot; generated {datetime.now():%d %b %Y %H:%M}</div></header>
<main>
<div class="kpis">
<div class="kpi"><div class="v">{total}</div><div class="l">Students</div></div>
<div class="kpi"><div class="v">{avg_g3:.2f}</div><div class="l">Average final mark (G3 / 20)</div></div>
<div class="kpi"><div class="v">{100 * pass_n / total:.1f}%</div><div class="l">Passing (G3 &ge; 10)</div></div>
<div class="kpi"><div class="v">{risk_n}</div><div class="l">At-risk (G3 &lt; 10)</div></div>
<div class="kpi"><div class="v">{avg_abs:.1f}</div><div class="l">Average absences</div></div>
</div>
<div class="grid">
<div class="card"><h2>Students by performance category</h2>{bar_chart(cat_items, fmt="{:.0f}")}
<p class="note">Excellent: G3 &ge; 15 &middot; Pass: 10&ndash;14 &middot; At-Risk: below 10</p></div>
<div class="card"><h2>Distribution of final marks (G3)</h2>{column_chart(hist_items)}</div>
<div class="card"><h2>Average G3 by school</h2>{bar_chart(by_school)}</div>
<div class="card"><h2>Average G3 by sex</h2>{bar_chart(by_sex)}</div>
<div class="card"><h2>Average G3 by weekly study time</h2>{bar_chart(by_study)}
<p class="note">1: &lt;2h, 2: 2&ndash;5h, 3: 5&ndash;10h, 4: &gt;10h</p></div>
<div class="card"><h2>Average G3 by past class failures</h2>{bar_chart(by_fail)}</div>
{model_section(metrics)}
<div class="card" style="grid-column:1/-1"><h2>Ten lowest-scoring students (may need academic support)</h2>
{table(["School", "Sex", "Age", "Study time", "Failures", "Absences", "G1", "G2", "G3"], risk_rows)}
<p class="note">Averages show associations in this dataset, not causes.</p></div>
</div></main></body></html>"""

    out = OUTPUT_DIR / "dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print("Dashboard written to:", out)
    print("Open it with:  open", out)
    spark.stop()


if __name__ == "__main__":
    main()
