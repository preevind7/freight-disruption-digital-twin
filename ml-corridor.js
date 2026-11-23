(function () {
  "use strict";
  const FDT = window.FDT = window.FDT || {};

  // Screens that read corridor risk; they wait for the ML forecast before rendering.
  const PAGES_USING_FORECAST = ["dashboard", "risk-map", "segment", "simulator", "results", "assistant", "library", "vehicles", "trip", "departure", "hotspots", "alerts"];
  const forecasts = {};      // corridorId -> backend forecast
  const pending = {};        // corridorId -> Promise
  let seedsRemoved = false;

  function istHour() {
    return Number(new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", hour12: false }).format(new Date())) % 24;
  }
  function hhmm(iso) {
    return new Intl.DateTimeFormat("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: false }).format(new Date(iso));
  }
  function statusFromRisk(risk) {
    return risk >= 0.6 ? "At risk" : risk >= 0.3 ? "Watch" : "On track";
  }

  // Replace the prototype's dummy numbers with the ML forecast, in place, so every screen uses it.
  function apply(forecast) {
    const hour = istHour();
    forecast.segments.forEach(function (item) {
      const segment = FDT.data.segmentById[item.id];
      if (!segment) return;
      segment.risk = item.hourly.risk.slice();
      // Rainfall is a DAILY forecast: the same expected mm/day for every hour.
      segment.rainMmHr = Array.from({ length: 24 }, function () { return item.rainfall.expected_mm; });
      segment.rainProbability = Array.from({ length: 24 }, function () { return item.rainfall.p_rain; });
      segment.floodThresholdMmHr = 64.5; // IMD heavy rain, mm/day
      segment.ml = item;
    });
    // Alerts for this corridor now come from the models.
    FDT.data.alerts = FDT.data.alerts.filter(function (alert) { return alert.corridorId !== forecast.corridor_id && !String(alert.id).startsWith("ALT-"); })
      .concat(forecast.alerts.map(function (alert) {
        return {
          id: alert.id, severity: alert.severity, title: alert.title, segmentId: alert.segment_id, corridorId: forecast.corridor_id,
          time: alert.hour === null || alert.hour === undefined ? hhmm(forecast.generated_at) : String(alert.hour).padStart(2, "0") + ":00",
          read: false, closed: false, ml: true, type: alert.type
        };
      }));
    // Trip status follows the ML risk of the segment each vehicle is on.
    FDT.data.trips.forEach(function (trip) {
      if (trip.completed || trip.corridorId !== forecast.corridor_id) return;
      const segment = FDT.data.segmentById[trip.currentSegment];
      if (segment && segment.ml) {
        trip.status = statusFromRisk(segment.risk[hour]);
        trip.reason = segment.id + " risk " + segment.risk[hour].toFixed(2) + " at " + String(hour).padStart(2, "0") + ":00 (ML forecast)";
      }
    });
    // Remove the prototype's hand-written example scenarios (their results were not computed).
    if (!seedsRemoved) { FDT.data.scenarios = []; seedsRemoved = true; }
    forecasts[forecast.corridor_id] = forecast;
    updateBadge();
    if (FDT.updateAlertCounts) FDT.updateAlertCounts();
  }

  function updateBadge() {
    const badge = document.querySelector(".topbar__badge-optional");
    const forecast = forecasts[FDT.state.corridorId];
    if (!badge || !forecast) return;
    const sources = Object.keys(forecast.rainfall_sources).filter(function (key) { return forecast.rainfall_sources[key]; });
    const ts = forecast.traffic_sources || { live: 0 };
    const traffic = ts.live ? (ts.model ? "live " + ts.live + "/" + (ts.live + ts.model) : "live") : "model";
    badge.innerHTML = "ML " + String(istHour()).padStart(2, "0") + ":00 <span class=\"simulated\">rain: " + FDT.escapeHtml(sources.join(" + ")) + "</span> <span class=\"simulated\">traffic: " + FDT.escapeHtml(traffic) + "</span>";
    badge.title = "Forecast for " + forecast.date + " · generated " + hhmm(forecast.generated_at) + " IST" + (forecast.live_traffic_status ? " · traffic: " + forecast.live_traffic_status : "");
  }

  function load(corridorId, refresh) {
    if (pending[corridorId] && !refresh) return pending[corridorId];
    const query = "?corridor_id=" + encodeURIComponent(corridorId) + (FDT.state.forecastDate ? "&date=" + encodeURIComponent(FDT.state.forecastDate) : "") + (refresh ? "&refresh=true" : "");
    pending[corridorId] = FDT.api.get("/api/corridor/forecast" + query).then(function (forecast) {
      apply(forecast);
      return forecast;
    }).catch(function (error) {
      delete pending[corridorId];
      throw error;
    });
    return pending[corridorId];
  }

  FDT.mlCorridor = {
    hour: istHour,
    needsForecast: function (page) { return PAGES_USING_FORECAST.includes(page); },
    get: function (corridorId) { return forecasts[corridorId || FDT.state.corridorId] || null; },
    ensure: function (corridorId) { return forecasts[corridorId] ? Promise.resolve(forecasts[corridorId]) : load(corridorId); },
    refresh: function (corridorId) { return load(corridorId || FDT.state.corridorId, true); },
    updateBadge: updateBadge,
    reset: function () { Object.keys(forecasts).forEach(function (k) { delete forecasts[k]; }); Object.keys(pending).forEach(function (k) { delete pending[k]; }); },
    // Plain-language breakdown of one segment-hour, shared by the segment page and the assistant.
    explain: function (segment, hour) {
      const ml = segment && segment.ml;
      const forecast = forecasts[segment.corridorId];
      if (!ml || !forecast) return null;
      const w = forecast.method.weights;
      return {
        risk: ml.hourly.risk[hour],
        rain: { value: w.rain * ml.rainfall.rain_component, pHeavy: ml.rainfall.p_heavy, threshold: ml.rainfall.alert_threshold !== undefined ? ml.rainfall.alert_threshold : forecast.heavy_alert_threshold, rule: ml.rainfall.alert_rule || forecast.heavy_alert_rule, forecastMm: ml.rainfall.forecast_mm, modelUsed: ml.rainfall.model_used || "", alert: ml.rainfall.heavy_alert, reason: ml.rainfall.alert_reason || "", expectedMm: ml.rainfall.expected_mm, category: ml.rainfall.predicted_category, source: ml.rainfall.source, sourceDetail: ml.rainfall.source_detail, city: ml.cities.rainfall },
        accident: { value: w.accident * ml.hourly.accident_risk[hour], score: ml.hourly.accident_risk[hour], weather: ml.conditions.weather, visibility: ml.conditions.visibility },
        congestion: { value: w.congestion * ml.hourly.p_high_congestion[hour], pHigh: ml.hourly.p_high_congestion[hour], level: ml.hourly.congestion_level[hour], city: ml.cities.road },
        event: forecast.event,
        formula: forecast.method.formula
      };
    }
  };
}());
