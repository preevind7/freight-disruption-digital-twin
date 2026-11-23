(function () {
  "use strict";
  const FDT = window.FDT = window.FDT || {};
  FDT.pages = FDT.pages || {};
  const TITLE = "Accident Risk";
  const FACTOR_LABELS = {
    city: "City", road_type: "Road type", weather: "Weather", visibility: "Visibility", traffic_density: "Traffic density",
    festival: "Festival", day_of_week: "Day of week", lanes: "Lanes", traffic_signal: "Traffic signal", temperature: "Temperature",
    hour_sin: "Hour of day", hour_cos: "Hour of day", is_peak_hour: "Peak hour", is_weekend: "Weekend"
  };

  function renderForm(root, meta) {
    const ml = FDT.ml;
    const o = meta.input_options;
    const now = new Date();
    const dayName = now.toLocaleDateString("en-US", { weekday: "long" });
    root.innerHTML =
      '<header class="page-heading"><div><p class="small muted">ML PREDICTIONS · MODEL 2 · <span class="simulated">' + (FDT.demo ? "Saved sample · demo" : "Live backend") + '</span></p><h1>' + TITLE + '</h1><p>Score how risky the road, weather, traffic and time conditions are, on a 0–1 scale learned from 20,000 Indian road-accident records.</p></div><div class="page-actions"><a class="button" href="#/model">Model & Validation</a></div></header>' +
      '<div class="ml-layout"><form id="acc-form" class="card ml-form" novalidate>' +
      '<div class="card__header"><div><h2>Trip conditions</h2><p>Values available in the training data (' + ml.esc(meta.dataset) + ').</p></div></div>' +
      '<div class="field-row"><div class="field"><label for="acc-city">City</label><select id="acc-city">' + ml.options(o.city, "Mumbai") + '</select></div>' +
      '<div class="field"><label for="acc-road">Road type</label><select id="acc-road">' + ml.options(o.road_type, "highway") + '</select></div></div>' +
      '<div class="field-row"><div class="field"><label for="acc-weather">Weather</label><select id="acc-weather">' + ml.options(o.weather, "rain") + '</select></div>' +
      '<div class="field"><label for="acc-visibility">Visibility</label><select id="acc-visibility"></select><small class="field__hint">Only combinations seen in the data are offered.</small></div></div>' +
      '<div class="field-row"><div class="field"><label for="acc-density">Traffic density</label><select id="acc-density">' + ml.options(o.traffic_density, "high") + '</select></div>' +
      '<div class="field"><label for="acc-temp">Temperature (°C)</label><input id="acc-temp" type="number" min="-10" max="55" step="1" value="28" required><small class="field__hint">Training range ' + o.temperature[0] + '–' + o.temperature[1] + ' °C</small></div></div>' +
      '<div class="field-row"><div class="field"><label for="acc-hour">Hour of travel (0–23)</label><input id="acc-hour" type="number" min="0" max="23" step="1" value="' + now.getHours() + '" required></div>' +
      '<div class="field"><label for="acc-day">Day of week</label><select id="acc-day">' + ml.options(o.day_of_week, dayName) + '</select></div></div>' +
      '<div class="field-row"><div class="field"><label for="acc-lanes">Lanes</label><input id="acc-lanes" type="number" min="' + o.lanes[0] + '" max="' + o.lanes[1] + '" step="1" value="4" required></div>' +
      '<div class="field"><label for="acc-festival">Festival</label><select id="acc-festival">' + ml.options(o.festival, "None") + '</select></div></div>' +
      '<label class="check-option"><input id="acc-signal" type="checkbox"><span><strong>Traffic signal at location</strong><small>Tick if the road has a signalised junction.</small></span></label>' +
      '<div class="ml-actions"><button type="submit" class="button button--primary" id="acc-submit">Predict accident risk</button></div>' +
      '</form><section id="acc-result" class="ml-result" aria-live="polite"><div class="card ml-placeholder"><h2>Prediction</h2><p class="muted">Choose the conditions and select <strong>Predict accident risk</strong>.</p></div></section></div>';

    const weather = root.querySelector("#acc-weather");
    const visibility = root.querySelector("#acc-visibility");
    function syncVisibility() {
      const allowed = o.weather_visibility_pairs.filter(function (pair) { return pair.split("|")[0] === weather.value; }).map(function (pair) { return pair.split("|")[1]; });
      const current = visibility.value;
      visibility.innerHTML = ml.options(allowed, allowed.includes(current) ? current : allowed[0]);
    }
    weather.addEventListener("change", syncVisibility);
    syncVisibility();

    const form = root.querySelector("#acc-form");
    const result = root.querySelector("#acc-result");
    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      if (!form.reportValidity()) return;
      const button = root.querySelector("#acc-submit");
      const payload = {
        city: root.querySelector("#acc-city").value,
        road_type: root.querySelector("#acc-road").value,
        lanes: Number(root.querySelector("#acc-lanes").value),
        traffic_signal: root.querySelector("#acc-signal").checked,
        weather: weather.value,
        visibility: visibility.value,
        temperature: Number(root.querySelector("#acc-temp").value),
        traffic_density: root.querySelector("#acc-density").value,
        hour: Number(root.querySelector("#acc-hour").value),
        day_of_week: root.querySelector("#acc-day").value,
        festival: root.querySelector("#acc-festival").value
      };
      ml.loading(button, true);
      try {
        const data = await FDT.api.post("/api/predict/accident", payload);
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
    const kind = ml.levelKind(data.risk_level);
    const drivers = data.drivers.map(function (d) {
      const value = d.factor === "is_peak_hour" ? (d.value === "1" ? "yes" : "no") : d.value;
      return { label: (FACTOR_LABELS[d.factor] || d.factor) + ": " + value, value: d.effect, text: (d.effect > 0 ? "+" : "") + d.effect.toFixed(2), kind: d.effect > 0 ? "alert" : "safe" };
    });
    container.innerHTML =
      (data.risk_level === "High" ? '<section class="risk-warning" role="status"><span class="risk-warning__icon" aria-hidden="true">!</span><div><strong>High accident risk.</strong> These conditions are among the riskiest in the training data.</div></section>' : "") +
      '<div class="kpi-grid ml-kpis ml-kpis--3">' +
      '<article class="kpi-card kpi-card--risk"><span class="kpi-card__label">Risk score <small>0 – 1</small></span><strong class="kpi-card__value number">' + data.risk_score.toFixed(2) + '</strong><span class="pill pill--' + kind + '">' + ml.esc(data.risk_level) + ' risk</span></article>' +
      '<article class="kpi-card"><span class="kpi-card__label">Uncertainty</span><strong class="kpi-card__value number">± ' + data.uncertainty.toFixed(2) + '</strong><span class="kpi-card__note">90% of validation errors were within this</span></article>' +
      '<article class="kpi-card"><span class="kpi-card__label">Peak hour</span><strong class="kpi-card__value">' + (data.is_peak_hour ? "Yes" : "No") + '</strong><span class="kpi-card__note">Peak: 08–10 and 17–19</span></article></div>' +
      '<section class="card"><div class="card__header"><div><h2>What drives this score</h2><p>Change versus average conditions in the data · red raises risk, green lowers it</p></div></div>' +
      (drivers.length ? ml.bars(drivers) : '<p class="muted">Explanations are available when the selected model is linear.</p>') +
      '<p class="ml-note">' + ml.esc(data.note) + '</p>' + ml.modelLine(data.model) + '</section>';
  }

  function render(root) {
    FDT.ml.skeleton(root, TITLE);
    FDT.api.models().then(function (models) {
      if (!root.isConnected) return;
      const meta = models.accident;
      if (!meta.status.loaded) throw Object.assign(new Error(meta.status.error), { status: 503 });
      renderForm(root, meta);
    }).catch(function (error) {
      if (!root.isConnected) return;
      FDT.ml.unavailable(root, TITLE, error, function () { FDT.api.models(true); render(root); });
    });
  }

  FDT.pages["predict-accident"] = { title: TITLE, render: render, destroy: function () {} };
}());
