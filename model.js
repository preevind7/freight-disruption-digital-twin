(function () {
  "use strict";
  const FDT = window.FDT = window.FDT || {};
  FDT.pages = FDT.pages || {};
  const TITLE = "Model & Validation";

  const MODELS = [
    { key: "rainfall", title: "Model 1 · Rainfall & flood-trigger risk", page: "/predict/rainfall", baselineKey: "test_baselines" },
    { key: "accident", title: "Model 2 · Accident risk", page: "/predict/accident", baselineKey: "test_baseline_mean" },
    { key: "traffic", title: "Model 3 · Festival traffic congestion", page: "/predict/traffic", baselineKey: "test_baselines" }
  ];
  const METRIC_LABELS = {
    log_loss: "Log loss ↓", macro_f1: "Macro-F1 ↑", accuracy: "Accuracy ↑", balanced_accuracy: "Balanced acc. ↑",
    heavy_pr_auc: "Heavy-rain PR-AUC ↑", heavy_roc_auc: "Heavy-rain ROC-AUC ↑", mae: "MAE ↓", rmse: "RMSE ↓", r2: "R² ↑", risk_band_accuracy: "Risk-band acc. ↑"
  };

  function fmt(value) {
    return value === undefined || value === null ? "—" : Number(value).toFixed(3);
  }
  function metricTable(rows, keys) {
    const ml = FDT.ml;
    return '<div class="table-wrap"><table class="data-table ml-metric-table"><thead><tr><th scope="col">Model</th>' +
      keys.map(function (k) { return '<th scope="col">' + ml.esc(METRIC_LABELS[k] || k) + "</th>"; }).join("") + "</tr></thead><tbody>" +
      rows.map(function (row) {
        return "<tr" + (row.highlight ? ' class="ml-row--selected"' : "") + '><td data-label="Model">' + ml.esc(row.name) + (row.highlight ? ' <span class="pill pill--info">selected</span>' : "") + "</td>" +
          keys.map(function (k) { return '<td class="number" data-label="' + ml.esc(METRIC_LABELS[k] || k) + '">' + fmt(row.metrics[k]) + "</td>"; }).join("") + "</tr>";
      }).join("") + "</tbody></table></div>";
  }

  function modelCard(spec, entry) {
    const ml = FDT.ml;
    if (!entry || !entry.report) {
      const message = entry && entry.status && entry.status.error ? entry.status.error : "No evaluation report found. Train the model first.";
      return '<section class="card"><div class="card__header"><div><h2>' + ml.esc(spec.title) + "</h2></div></div>" + ml.errorBox({ status: 503, message: message }) + "</section>";
    }
    const r = entry.report;
    const keys = Object.keys(r.test_metrics);
    const baselines = r[spec.baselineKey] && r[spec.baselineKey].mae !== undefined ? { "training mean": r[spec.baselineKey] } : (r[spec.baselineKey] || {});
    const testRows = [{ name: r.selected_model, metrics: r.test_metrics, highlight: true }].concat(Object.keys(baselines).map(function (name) { return { name: "Baseline · " + name, metrics: baselines[name] }; }));
    const valRows = Object.keys(r.validation_comparison).map(function (name) { return { name: name, metrics: r.validation_comparison[name], highlight: name === r.selected_model }; });
    const s = r.split;
    let extra = "";
    if (r.test_heavy_alert) {
      const a = r.test_heavy_alert;
      extra = '<p class="small">Heavy-rain alert (' + ml.esc(r.heavy_alert_rule ? "rule chosen on validation: " + r.heavy_alert_rule_text + ((r.heavy_alert_rule === "model" || r.heavy_alert_rule === "model_or_persistence") ? ", threshold " + ml.percent(r.heavy_alert_threshold) : "") : "threshold " + ml.percent(r.heavy_alert_threshold)) + '): precision ' + fmt(a.precision) + ", recall " + fmt(a.recall) + " — " + a.alerts_raised + " alerts for " + a.heavy_days + " actual heavy-rain days in the test period.</p>";
    }
    if (r.prediction_error_p90 !== undefined) extra = '<p class="small">90% of validation predictions were within ±' + fmt(r.prediction_error_p90) + " of the true score.</p>";
    return '<section class="card ml-model-card"><div class="card__header"><div><h2>' + ml.esc(spec.title) + "</h2><p>" + ml.esc(r.task) + '</p></div><a class="button" href="#' + spec.page + '">Try it</a></div>' +
      '<dl class="ml-facts"><div><dt>Algorithm</dt><dd>' + ml.esc(r.selected_model) + "</dd></div><div><dt>Dataset</dt><dd>" + ml.esc(r.dataset_description ? r.dataset_description.replace(/`/g, "") : [].concat(r.dataset).join(" + ")) + "</dd></div>" +
      "<div><dt>Train</dt><dd>" + ml.esc(s.train.from + " → " + s.train.to) + " · " + s.train.rows.toLocaleString("en-IN") + " rows</dd></div>" +
      "<div><dt>Validation</dt><dd>" + ml.esc(s.validation.from + " → " + s.validation.to) + " · " + s.validation.rows.toLocaleString("en-IN") + " rows</dd></div>" +
      "<div><dt>Test</dt><dd>" + ml.esc(s.test.from + " → " + s.test.to) + " · " + s.test.rows.toLocaleString("en-IN") + " rows</dd></div>" +
      "<div><dt>Selected by</dt><dd>" + ml.esc(r.selection_metric) + "</dd></div></dl>" +
      "<h3>Test period · model vs baselines</h3>" + metricTable(testRows, keys) + extra +
      '<details class="ml-details"><summary>Algorithm comparison on the validation period</summary>' + metricTable(valRows, Object.keys(r.validation_comparison[r.selected_model])) + "</details>" +
      '<h3>Limitations</h3><ul class="ml-limitations">' + r.limitations.map(function (item) { return "<li>" + ml.esc(item) + "</li>"; }).join("") + "</ul></section>";
  }

  function render(root) {
    const ml = FDT.ml;
    ml.skeleton(root, TITLE);
    FDT.api.models(true).then(function (models) {
      if (!root.isConnected) return;
      root.innerHTML =
        '<header class="page-heading"><div><p class="small muted">FLEET & PLANNING · <span class="simulated">' + (FDT.demo ? "Saved sample · demo" : "Live backend") + '</span></p><h1>' + TITLE + '</h1><p>Evaluation of the three ML models on a held-out, later time period. Every number below is read from the training reports on the backend.</p></div><div class="page-actions"><span class="pill pill--neutral">' + (FDT.demo ? "Demo: saved from the backend" : "API: " + ml.esc(FDT.api.baseUrl())) + "</span></div></header>" +
        '<div class="departure-disclaimer" role="note"><strong>How to read this</strong><span>Each model is trained on earlier dates, tuned on a validation period and scored once on the most recent dates. ↑ means higher is better, ↓ lower is better. Baselines are simple rules the model should beat.</span></div>' +
        MODELS.map(function (spec) { return modelCard(spec, models[spec.key]); }).join("");
    }).catch(function (error) {
      if (!root.isConnected) return;
      ml.unavailable(root, TITLE, error, function () { render(root); });
    });
  }

  FDT.pages.model = { title: TITLE, render: render, destroy: function () {} };
}());
