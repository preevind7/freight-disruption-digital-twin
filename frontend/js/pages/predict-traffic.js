(function () {
  "use strict";
  const FDT = window.FDT = window.FDT || {};
  FDT.pages = FDT.pages || {};
  const TITLE = "Festival Traffic Congestion";
  const AUTO = "__auto__";

  function today() {
    const now = new Date();
    return new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  }

  function renderForm(root, meta) {
    const ml = FDT.ml;
    const o = meta.input_options;
    const festivalLabels = { [AUTO]: "Auto-detect from event calendar" };
    root.innerHTML =
      '<header class="page-heading"><div><p class="small muted">ML PREDICTIONS · MODEL 3 · <span class="simulated">Live backend</span></p><h1>' + TITLE + '</h1><p>Predict the traffic congestion level for a city, date and hour, with festivals detected from the significant-dates calendar.</p></div><div class="page-actions"><a class="button" href="#/model">Model & Validation</a></div></header>' +
      '<div class="ml-layout"><form id="traffic-form" class="card ml-form" novalidate>' +
      '<div class="card__header"><div><h2>When and where</h2><p>Trained on ' + ml.esc([].concat(meta.dataset).join(" + ")) + '.</p></div></div>' +
      '<div class="field-row"><div class="field"><label for="tr-city">City</label><select id="tr-city">' + ml.options(o.city, "Bangalore") + '</select></div>' +
      '<div class="field"><label for="tr-date">Date</label><input id="tr-date" type="date" required value="' + today() + '"></div></div>' +
      '<div class="field-row"><div class="field"><label for="tr-hour">Hour (0–23)</label><input id="tr-hour" type="number" min="0" max="23" step="1" required value="18"></div>' +
      '<div class="field"><label for="tr-festival">Festival / event</label><select id="tr-festival">' + ml.options([AUTO].concat(o.festival), AUTO, festivalLabels) + '</select><small class="field__hint">Auto-detect looks the date up in significant_dates.csv.</small></div></div>' +
      '<div class="field-row"><div class="field"><label for="tr-road">Road type</label><select id="tr-road">' + ml.options(o.road_type, "urban") + '</select></div>' +
      '<div class="field"><label for="tr-weather">Weather</label><select id="tr-weather">' + ml.options(o.weather, "clear") + '</select></div></div>' +
      '<div class="field-row"><div class="field"><label for="tr-lanes">Lanes</label><input id="tr-lanes" type="number" min="' + o.lanes[0] + '" max="' + o.lanes[1] + '" step="1" required value="3"></div><div></div></div>' +
      '<label class="check-option"><input id="tr-signal" type="checkbox" checked><span><strong>Traffic signal at location</strong><small>Tick if the road has a signalised junction.</small></span></label>' +
      '<div class="ml-actions"><button type="submit" class="button button--primary" id="tr-submit">Predict congestion</button></div>' +
      '</form><section id="traffic-result" class="ml-result" aria-live="polite"><div class="card ml-placeholder"><h2>Prediction</h2><p class="muted">Pick a city, date and hour, then select <strong>Predict congestion</strong>. Try a festival date such as 2026-11-08 (Deepavali).</p></div></section></div>';

    const form = root.querySelector("#traffic-form");
    const result = root.querySelector("#traffic-result");
    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      if (!form.reportValidity()) return;
      const button = root.querySelector("#tr-submit");
      const festival = root.querySelector("#tr-festival").value;
      const payload = {
        city: root.querySelector("#tr-city").value,
        date: root.querySelector("#tr-date").value,
        hour: Number(root.querySelector("#tr-hour").value),
        road_type: root.querySelector("#tr-road").value,
        lanes: Number(root.querySelector("#tr-lanes").value),
        traffic_signal: root.querySelector("#tr-signal").checked,
        weather: root.querySelector("#tr-weather").value,
        festival: festival === AUTO ? null : festival
      };
      ml.loading(button, true);
      try {
        const data = await FDT.api.post("/api/predict/traffic", payload);
        if (!root.isConnected) return;
        renderResult(result, data);
      } catch (error) {
        result.innerHTML = ml.errorBox(error);
      } finally {
        ml.loading(button, false);
      }
    });
  }

  function renderResult(container, data) {
    const ml = FDT.ml;
    const level = data.congestion_level;
    const kind = ml.levelKind(level);
    const label = level.charAt(0).toUpperCase() + level.slice(1);
    const event = data.festival_event === "None" ? "None" : data.festival_event;
    container.innerHTML =
      '<div class="kpi-grid ml-kpis ml-kpis--3">' +
      '<article class="kpi-card kpi-card--risk"><span class="kpi-card__label">Congestion level</span><strong class="kpi-card__value">' + ml.esc(label) + '</strong><span class="pill pill--' + kind + '">' + ml.esc(label) + ' traffic</span></article>' +
      '<article class="kpi-card"><span class="kpi-card__label">Confidence</span><strong class="kpi-card__value number">' + ml.percent(data.confidence, 0) + '</strong><span class="kpi-card__note">Probability of this level</span></article>' +
      '<article class="kpi-card"><span class="kpi-card__label">Festival / event</span><strong class="kpi-card__value ml-kpi-text">' + ml.esc(event) + '</strong><span class="kpi-card__note">' + (data.calendar_event ? "Calendar: " + ml.esc(data.calendar_event) : "No event in the calendar") + (data.is_peak_hour ? " · peak hour" : "") + '</span></article></div>' +
      '<section class="card"><div class="card__header"><div><h2>Probability of each congestion level</h2><p>The model is uncertain when the bars are close; treat close calls with care.</p></div></div>' +
      ml.bars(["low", "medium", "high"].map(function (name) {
        return { label: name.charAt(0).toUpperCase() + name.slice(1), value: data.probabilities[name], text: ml.percent(data.probabilities[name]), kind: name === "high" ? "alert" : name === "medium" ? "warn" : "safe" };
      }), { max: 1 }) +
      '<p class="ml-note">' + ml.esc(data.note) + '</p>' + ml.modelLine(data.model) + '</section>';
  }

  function render(root) {
    FDT.ml.skeleton(root, TITLE);
    FDT.api.models().then(function (models) {
      if (!root.isConnected) return;
      const meta = models.traffic;
      if (!meta.status.loaded) throw Object.assign(new Error(meta.status.error), { status: 503 });
      renderForm(root, meta);
    }).catch(function (error) {
      if (!root.isConnected) return;
      FDT.ml.unavailable(root, TITLE, error, function () { FDT.api.models(true); render(root); });
    });
  }

  FDT.pages["predict-traffic"] = { title: TITLE, render: render, destroy: function () {} };
}());
