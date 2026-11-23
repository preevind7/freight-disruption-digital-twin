(function () {
  "use strict";
  const FDT = window.FDT = window.FDT || {};
  FDT.pages = FDT.pages || {};
  const TITLE = "Rainfall & Flood Risk";

  function today() {
    const now = new Date();
    return new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  }

  function renderForm(root, meta, params) {
    const ml = FDT.ml;
    const cities = meta.input_options.location;
    const fallback = cities.find(function (c) { return c === "Mumbai" || c.indexOf("S03 ") === 0; }) || cities[0];
    const city = cities.includes(params.city) ? params.city : fallback;
    const perSegment = /open_meteo/.test(meta.dataset || "");
    const report = meta.report || {};
    root.innerHTML =
      '<header class="page-heading"><div><p class="small muted">ML PREDICTIONS · MODEL 1 · <span class="simulated">Live backend</span></p><h1>' + TITLE + '</h1><p>Forecast tomorrow\'s rainfall category and the chance of heavy rain (≥ 64.5 mm/day) — the flood trigger — from the last 7 days of rain.</p></div><div class="page-actions"><a class="button" href="#/model">Model & Validation</a></div></header>' +
      '<div class="ml-layout"><form id="rain-form" class="card ml-form" novalidate>' +
      '<div class="card__header"><div><h2>Inputs</h2><p>' + (perSegment ? "Real daily rainfall at each corridor segment (Open-Meteo / ERA5), as in the training data (" : "City-level daily rainfall, as in the training data (") + ml.esc(meta.dataset) + ').</p></div></div>' +
      '<div class="field-row"><div class="field"><label for="rain-city">' + (perSegment ? "Corridor segment" : "City") + '</label><select id="rain-city" required>' + ml.options(cities, city) + '</select></div>' +
      '<div class="field"><label for="rain-date">Forecast date</label><input id="rain-date" type="date" required value="' + ml.esc(params.date || today()) + '"></div></div>' +
      '<fieldset class="ml-days"><legend>Observed rainfall on the 7 days before the forecast date (mm)</legend><div class="ml-days__grid">' +
      Array.from({ length: 7 }, function (_, index) {
        return '<div class="field"><label for="rain-d' + (index + 1) + '">Day −' + (index + 1) + '</label><input id="rain-d' + (index + 1) + '" type="number" min="0" max="1000" step="any" required value="0"></div>';
      }).join("") +
      '</div><small class="field__hint">Day −1 is the day before the forecast date. Use “Fill from dataset” for any date between ' + ml.esc((report.split && report.split.train.from) || "2016-01-08") + ' and ' + ml.esc((report.split && report.split.test.to) || "2025-12-31") + '.</small></fieldset>' +
      (report.uses_forecast ? '<div class="field"><label for="rain-forecast">Rain forecast for the forecast date, made the day before (mm) · optional</label><input id="rain-forecast" type="number" min="0" max="1000" step="any" placeholder="leave empty if unknown"><small class="field__hint">With a forecast the model that learned from forecasts is used; without one, the past-rain-only model.</small></div>' : "") +
      '<div id="rain-history-status" class="small muted" aria-live="polite"></div>' +
      '<div class="ml-actions"><button type="button" class="button" id="rain-fill">Fill from dataset</button><button type="submit" class="button button--primary" id="rain-submit">Predict rainfall risk</button></div>' +
      '</form><section id="rain-result" class="ml-result" aria-live="polite"><div class="card ml-placeholder"><h2>Prediction</h2><p class="muted">Fill in the inputs and select <strong>Predict rainfall risk</strong>. Results come from the trained model on the backend.</p></div></section></div>';

    const form = root.querySelector("#rain-form");
    const result = root.querySelector("#rain-result");
    const status = root.querySelector("#rain-history-status");
    let observed = null; // actual outcome from the dataset, if the inputs were filled from it

    function values() {
      return Array.from({ length: 7 }, function (_, index) { return Number(root.querySelector("#rain-d" + (index + 1)).value); });
    }
    form.addEventListener("input", function (event) { if (event.target.id !== "rain-city") observed = null; });

    root.querySelector("#rain-fill").addEventListener("click", async function (event) {
      const button = event.currentTarget;
      const cityValue = root.querySelector("#rain-city").value;
      const dateValue = root.querySelector("#rain-date").value;
      if (!dateValue) { root.querySelector("#rain-date").reportValidity(); return; }
      ml.loading(button, true, "Loading…");
      status.textContent = "";
      try {
        const history = await FDT.api.get("/api/rainfall/history?city=" + encodeURIComponent(cityValue) + "&forecast_date=" + encodeURIComponent(dateValue));
        history.rainfall_last_7_days_mm.forEach(function (value, index) { root.querySelector("#rain-d" + (index + 1)).value = value; });
        const fcInput = root.querySelector("#rain-forecast");
        if (fcInput) fcInput.value = history.forecast_rain_mm === null || history.forecast_rain_mm === undefined ? "" : history.forecast_rain_mm;
        observed = history.actual_rainfall_mm === null ? null : history;
        status.textContent = "Filled with observed rainfall from " + history.source + (observed ? ". Actual rainfall on that day is known, so you can compare it with the forecast." : ".");
      } catch (error) {
        status.innerHTML = ml.errorBox(error);
      } finally {
        ml.loading(button, false);
      }
    });

    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      if (!form.reportValidity()) return;
      const button = root.querySelector("#rain-submit");
      const payload = { city: root.querySelector("#rain-city").value, forecast_date: root.querySelector("#rain-date").value, rainfall_last_7_days_mm: values() };
      const fcField = root.querySelector("#rain-forecast");
      if (fcField && fcField.value.trim() !== "") payload.forecast_rain_mm = Number(fcField.value);
      const observedNow = observed && observed.city === payload.city && observed.forecast_date === payload.forecast_date ? observed : null;
      ml.loading(button, true);
      result.setAttribute("aria-busy", "true");
      try {
        const data = await FDT.api.post("/api/predict/rainfall", payload);
        if (!root.isConnected) return;
        renderResult(result, data, observedNow);
      } catch (error) {
        result.innerHTML = ml.errorBox(error);
      } finally {
        ml.loading(button, false);
        result.removeAttribute("aria-busy");
      }
    });
  }

  function renderResult(container, data, observed) {
    const ml = FDT.ml;
    const kind = ml.levelKind(data.flood_trigger_risk_level);
    const heavyIndex = data.category_probabilities.length - 1;
    container.innerHTML =
      (data.heavy_rain_alert ? '<section class="risk-warning" role="status"><span class="risk-warning__icon" aria-hidden="true">!</span><div><strong>Heavy-rain alert.</strong> ' + ml.esc(data.city) + ': ' + ml.esc(data.alert_reason || "chance of ≥ 64.5 mm is above the alert threshold") + '. Expect possible waterlogging and road disruption.</div></section>' : "") +
      '<div class="kpi-grid ml-kpis">' +
      '<article class="kpi-card kpi-card--risk"><span class="kpi-card__label">Flood-trigger risk</span><strong class="kpi-card__value">' + ml.esc(data.flood_trigger_risk_level) + '</strong><span class="pill pill--' + kind + '">' + ml.esc(data.flood_trigger_risk_level) + ' risk</span></article>' +
      '<article class="kpi-card"><span class="kpi-card__label">Heavy-rain chance</span><strong class="kpi-card__value number">' + ml.percent(data.heavy_rain_probability) + '</strong><span class="kpi-card__note">' + ml.esc(ml.alertRuleShort(data.alert_rule, data.alert_threshold).replace(/^a/, "A")) + '</span></article>' +
      '<article class="kpi-card"><span class="kpi-card__label">Usual for this month</span><strong class="kpi-card__value number">' + ml.percent(data.usual_heavy_rain_probability) + '</strong><span class="kpi-card__note">Historical share of heavy-rain days</span></article>' +
      '<article class="kpi-card"><span class="kpi-card__label">Most likely</span><strong class="kpi-card__value ml-kpi-text">' + ml.esc(data.predicted_category.split(" (")[0]) + '</strong><span class="kpi-card__note">' + ml.esc(data.city) + ' · ' + ml.esc(data.forecast_date) + '</span></article></div>' +
      '<section class="card"><div class="card__header"><div><h2>Rainfall category probabilities</h2><p>' + ml.esc(data.explanation) + '</p></div></div>' +
      ml.bars(data.category_probabilities.map(function (item, index) {
        return { label: item.category, value: item.probability, text: ml.percent(item.probability), kind: index === heavyIndex ? "alert" : "up" };
      }), { max: 1 }) +
      (observed ? '<p class="ml-observed"><strong>Observed in dataset:</strong> ' + ml.esc(observed.actual_rainfall_mm) + ' mm — ' + ml.esc(observed.actual_category) + '</p>' : "") +
      '<p class="ml-note">' + ml.esc(data.note) + '</p>' + ml.modelLine(data.model) + '</section>';
  }

  function render(root, params) {
    FDT.ml.skeleton(root, TITLE);
    FDT.api.models().then(function (models) {
      if (!root.isConnected) return;
      const meta = models.rainfall;
      if (!meta.status.loaded) throw Object.assign(new Error(meta.status.error), { status: 503 });
      renderForm(root, meta, params || {});
    }).catch(function (error) {
      if (!root.isConnected) return;
      FDT.ml.unavailable(root, TITLE, error, function () { FDT.api.models(true); render(root, params); });
    });
  }

  FDT.pages["predict-rainfall"] = { title: TITLE, render: render, destroy: function () {} };
}());
