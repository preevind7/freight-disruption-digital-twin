(function () {
  "use strict";
  const FDT = window.FDT = window.FDT || {};
  FDT.pages = FDT.pages || {};
  function riskKind(risk) {
    return risk >= 0.6 ? "high" : risk >= 0.3 ? "moderate" : "low";
  }
  function render(root, params) {
    const segment = FDT.data.segmentById[(params.id || "").toUpperCase()];
    if (!segment) {
      root.innerHTML = '<section class="error-state" role="alert"><h1>Segment not found</h1><p>We could not find that highway segment in the pilot corridors.</p><a class="button" href="#/risk-map">Back to Corridor Risk Map</a></section>';
      return;
    }
    const corridor = FDT.data.getCorridor(segment.corridorId);
    const hour = FDT.mlCorridor ? FDT.mlCorridor.hour() : 10;
    const hourLabel = String(hour).padStart(2, "0") + ":00";
    const why = FDT.mlCorridor ? FDT.mlCorridor.explain(segment, hour) : null;
    const ml = segment.ml;
    const risk = segment.risk[hour];
    const rain = segment.rainMmHr[hour];
    const rainProbability = Math.round(segment.rainProbability[hour] * 100);
    const watched = FDT.state.watchedSegments.includes(segment.id);
    const affectedTrips = FDT.data.getTrips(corridor.id).filter(function (trip) { return !trip.completed; }).filter(function (trip) { return trip.currentSegment === segment.id; }).map(function (trip) { return trip.id; });
    const trips = affectedTrips.map(function (id) { return FDT.data.trips.find(function (trip) { return trip.id === id; }); }).filter(Boolean);
    // Contribution of each ML model at this hour (combined with: risk = 1 - (1-a)(1-b)(1-c)).
    const contributionItems = why ? [
      { label: "Heavy rain (rainfall model)", value: why.rain.value },
      { label: "Accident risk (accident model)", value: why.accident.value },
      { label: "High congestion (traffic model)", value: why.congestion.value }
    ] : [];
    const contributionTotal = risk;
    const delayP50 = Math.round(35 + risk * 110);
    const delayP90 = Math.round(60 + risk * 150);

    root.innerHTML =
      '<header class="page-heading segment-heading"><div><a class="back-link" href="#/risk-map">← Corridor map</a><p class="small muted">' + FDT.escapeHtml(corridor.name) + ' · ' + segment.km + ' km · ML forecast for ' + hourLabel + ' IST</p><h1>' + FDT.escapeHtml(segment.id + " · " + segment.name) + '</h1><p>' + FDT.escapeHtml(corridor.highway) + ' · heavy rain = 64.5 mm/day or more (IMD)' + (ml ? (ml.cities.rainfall.km === 0 ? ' · rainfall: real data at this segment (Open-Meteo)' : ' · nearest data city: ' + FDT.escapeHtml(ml.cities.rainfall.name) + ' (' + ml.cities.rainfall.km + ' km)') : "") + '</p></div><div class="page-actions"><button type="button" class="button" id="watch-segment" aria-pressed="' + String(watched) + '">' + (watched ? "Stop watching" : "Watch") + '</button><a class="button" href="#/assistant?q=' + encodeURIComponent("Why is " + segment.id + " risky right now?") + '&segment=' + encodeURIComponent(segment.id) + '">Ask assistant</a><a class="button button--primary" href="#/simulator?segment=' + encodeURIComponent(segment.id) + '&trip=' + encodeURIComponent(trips[0] ? trips[0].id : "") + '">Simulate reroute</a></div></header>' +
      (risk >= 0.6 ? '<section class="risk-warning" role="status"><span class="risk-warning__icon" aria-hidden="true">!</span><div><strong>High disruption risk.</strong> ' + trips.length + ' active trips may cross this segment. Run a reroute simulation to compare options.</div></section>' : "") +
      '<section class="kpi-grid segment-kpi-grid" aria-label="Segment forecast"><article class="kpi-card"><span class="kpi-card__label">Disruption risk <small>combined index</small></span><strong class="kpi-card__value number">' + risk.toFixed(2) + '</strong><span class="pill pill--' + riskKind(risk) + '">' + (risk >= 0.6 ? "High" : risk >= 0.3 ? "Moderate" : "Low") + ' risk</span></article>' +
      '<article class="kpi-card"><span class="kpi-card__label">Chance of rain today <small>≥ 2.5 mm</small></span><strong class="kpi-card__value number">' + rainProbability + '%</strong><span class="kpi-card__note">Rainfall model · input: ' + FDT.escapeHtml(ml ? ml.rainfall.source : "n/a") + '</span></article>' +
      '<article class="kpi-card"><span class="kpi-card__label">Expected rainfall</span><strong class="kpi-card__value number">' + Math.round(rain) + ' <span class="kpi-unit">mm/day</span></strong><span class="kpi-card__note">Heavy-rain chance ' + (why ? (why.rain.pHeavy * 100).toFixed(1) : "–") + '% · ' + (why ? FDT.escapeHtml(FDT.ml.alertRuleShort(why.rain.rule, why.rain.threshold)) : "alert –") + '</span></article>' +
      '<article class="kpi-card"><span class="kpi-card__label">Delay if disrupted <span class="simulated">Estimate</span></span><strong class="kpi-card__value number">' + delayP50 + ' min</strong><span class="kpi-card__note">P50 · P90 ' + delayP90 + ' min · rule of thumb, no delay data</span></article></section>' +
      '<div class="segment-chart-grid"><section class="card"><div class="card__header"><div><h2>Risk by hour today</h2><p>Combined ML risk · dashed line is the high-risk threshold</p></div><span class="simulated">ML</span></div><div id="segment-risk-chart"></div></section>' +
      '<section class="card"><div class="card__header"><div><h2>Chance of high congestion by hour</h2><p>Traffic model · ' + FDT.escapeHtml(why && why.event.festival !== "None" ? "festival/event: " + (why.event.calendar_text || why.event.festival) : "no festival today") + '</p></div><span class="simulated">ML</span></div><div id="segment-rain-chart"></div></section></div>' +
      '<div class="segment-details-grid"><section class="card"><div class="card__header"><div><h2>What drives this score at ' + hourLabel + '</h2><p>Weighted model outputs, combined as risk = 1 − (1−a)(1−b)(1−c)</p></div><strong class="number">' + contributionTotal.toFixed(2) + ' risk</strong></div><div id="segment-contributions"></div></section>' +
      '<section class="card vehicle-impact-card"><div class="card__header"><div><h2>How each vehicle type is affected</h2><p>Higher flood tolerance means less expected disruption</p></div></div><div class="table-wrap"><table class="data-table"><thead><tr><th scope="col">Vehicle</th><th scope="col">Flood tolerance</th><th scope="col">Expected delay</th></tr></thead><tbody>' +
      FDT.data.vehicles.map(function (vehicle) {
        const delay = Math.round(delayP50 / vehicle.floodTolerance);
        return '<tr><td data-label="Vehicle">' + FDT.icon(vehicle.icon) + '<span>' + FDT.escapeHtml(vehicle.type) + '</span></td><td class="number" data-label="Flood tolerance">' + vehicle.floodTolerance.toFixed(2) + '×</td><td class="number" data-label="Expected delay">' + delay + ' min</td></tr>';
      }).join("") +
      '</tbody></table></div></section></div>' +
      (ml && why ? '<section class="card ml-inputs-card"><div class="card__header"><div><h2>Model inputs for this segment</h2><p>What each model received · ' + FDT.escapeHtml(why.formula) + '</p></div><a class="button" href="#/model">Model accuracy</a></div><dl class="ml-facts">' +
        '<div><dt>Rainfall input</dt><dd>' + FDT.escapeHtml(ml.rainfall.source_detail) + '</dd></div>' +
        '<div><dt>Last 7 days (mm, newest first)</dt><dd>' + FDT.escapeHtml(ml.rainfall.last_7_days_mm.join(", ")) + '</dd></div>' +
        '<div><dt>Rainfall forecast</dt><dd>' + FDT.escapeHtml(ml.rainfall.predicted_category) + '</dd></div>' +
        (ml.rainfall.forecast_source && ml.rainfall.forecast_source !== "unused" ? '<div><dt>Forecast made the day before</dt><dd>' + (ml.rainfall.forecast_mm === null ? "not available · " : ml.rainfall.forecast_mm + " mm · ") + FDT.escapeHtml(ml.rainfall.model_used) + '</dd></div>' : "") +
        (ml.cities.rainfall.km === 0 ? '<div><dt>Rainfall data</dt><dd>This segment\'s own location (Open-Meteo)</dd></div>' : '<div><dt>Rainfall model city</dt><dd>' + FDT.escapeHtml(ml.cities.rainfall.name) + ' · ' + ml.cities.rainfall.km + ' km away</dd></div>') +
        '<div><dt>Traffic &amp; accident model city</dt><dd>' + FDT.escapeHtml(ml.cities.road.name) + ' · ' + ml.cities.road.km + ' km away</dd></div>' +
        '<div><dt>Weather used</dt><dd>' + FDT.escapeHtml(ml.conditions.weather + ", " + ml.conditions.visibility + " visibility") + '</dd></div>' +
        '<div><dt>Traffic at ' + hourLabel + '</dt><dd>' + FDT.escapeHtml(why.congestion.level) + ' (P high ' + Math.round(why.congestion.pHigh * 100) + '%)</dd></div>' +
        (ml.live_traffic ? '<div><dt>Live traffic now (TomTom)</dt><dd>' + ml.live_traffic.current_speed_kmh + ' km/h vs ' + ml.live_traffic.free_flow_speed_kmh + ' km/h free flow · ' + FDT.escapeHtml(ml.live_traffic.level) + (ml.live_traffic.road_closure ? " · ROAD CLOSED" : "") + ' · confidence ' + Math.round(ml.live_traffic.confidence * 100) + '%</dd></div>' : "") +
        '<div><dt>Accident risk at ' + hourLabel + '</dt><dd>' + why.accident.score.toFixed(2) + '</dd></div>' +
        '<div><dt>Festival / event</dt><dd>' + FDT.escapeHtml(why.event.calendar_text || why.event.festival) + '</dd></div>' +
        '</dl></section>' : "") +
      '<section class="card segment-trips-card"><div class="card__header"><div><h2>Trips that will cross this segment</h2><p>' + trips.length + ' active trips in the synthetic corridor plan</p></div><a href="#/vehicles">All vehicles</a></div><div class="segment-trip-list">' +
      (trips.length ? trips.map(function (trip) {
        const vehicle = FDT.data.getVehicle(trip.vehicleId);
        const status = FDT.state.reroutedTrips.includes(trip.id) ? "Rerouted" : trip.status;
        const badge = status === "At risk" ? "danger" : status === "Watch" ? "warning" : status === "Rerouted" ? "info" : "success";
        return '<article class="segment-trip"><span class="trip-cell__icon">' + FDT.icon(vehicle.icon) + '</span><div class="segment-trip__main"><a href="#/trip/' + encodeURIComponent(trip.id) + '">' + FDT.escapeHtml(trip.id) + ' · ' + FDT.escapeHtml(trip.plate) + '</a><small>' + FDT.escapeHtml(trip.origin) + ' → ' + FDT.escapeHtml(trip.destination) + ' · now on ' + FDT.escapeHtml(trip.currentSegment) + '</small></div><span class="pill pill--' + badge + '">' + FDT.escapeHtml(status) + '</span><a class="button button--subtle" href="#/simulator?trip=' + encodeURIComponent(trip.id) + '&segment=' + encodeURIComponent(segment.id) + '">Simulate</a></article>';
      }).join("") : '<div class="empty-state"><h3>No trips to show</h3><p>There are no active trips on this corridor.</p></div>') +
      '</div></section>';

    root.querySelector("#segment-risk-chart").append(FDT.charts.line({ label: "Disruption risk over the next 24 hours", values: segment.risk, threshold: 0.6, color: "var(--blue)", min: 0, max: 1 }));
    root.querySelector("#segment-rain-chart").append(FDT.charts.line({ label: "Probability of high congestion by hour", values: ml ? ml.hourly.p_high_congestion : segment.risk, threshold: 0.5, color: "var(--amber)", min: 0, max: 1 }));
    root.querySelector("#segment-contributions").append(FDT.charts.contributions(contributionItems));
    const watchButton = root.querySelector("#watch-segment");
    watchButton.addEventListener("click", function () {
      const isWatching = FDT.state.watchedSegments.includes(segment.id);
      FDT.store.update(function (state) {
        state.watchedSegments = isWatching
          ? state.watchedSegments.filter(function (id) { return id !== segment.id; })
          : state.watchedSegments.concat(segment.id);
      });
      watchButton.setAttribute("aria-pressed", String(!isWatching));
      watchButton.textContent = isWatching ? "Watch" : "Stop watching";
      FDT.toast(isWatching ? segment.id + " removed from your watch list." : "Now watching " + segment.id + ".");
    });
    const note = FDT.ui.designNote(1, "The risk trend and factor bars show both the forecast and why it is elevated.");
    if (note) root.append(note);
  }

  FDT.pages.segment = {
    title: "Segment detail",
    render: render,
    destroy: function () {}
  };
}());
