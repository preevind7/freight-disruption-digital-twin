(function () {
  "use strict";
  const FDT = window.FDT = window.FDT || {};
  FDT.pages = FDT.pages || {};

  function pad(hour) { return String(hour).padStart(2, "0") + ":00"; }
  function band(risk) { return risk >= 0.6 ? "High" : risk >= 0.3 ? "Moderate" : "Low"; }
  function pct(value, digits) { return (value * 100).toFixed(digits === undefined ? 0 : digits) + "%"; }
  function findSegment(text, params) {
    const match = (params && params.segment) || (text.match(/\b([SC])\s?(\d{1,2})\b/i) || []).slice(1).join("");
    if (!match) return null;
    const id = match.charAt(0).toUpperCase() + String(Number(match.slice(1))).padStart(2, "0");
    return FDT.data.segmentById[id] || null;
  }
  function bestAndWorst(segment) {
    let best = 0, worst = 0;
    segment.risk.forEach(function (value, hour) {
      if (value < segment.risk[best]) best = hour;
      if (value > segment.risk[worst]) worst = hour;
    });
    return { best: best, worst: worst };
  }
  function topSegment(corridor, hour) {
    return FDT.data.getSegments(corridor.id).slice().sort(function (a, b) { return b.risk[hour] - a.risk[hour]; })[0];
  }

  function suggestionsFor(corridor) {
    const hour = FDT.mlCorridor.hour();
    const peak = topSegment(corridor, hour);
    const trip = FDT.data.getTrips(corridor.id).find(function (t) { return !t.completed && t.status === "At risk"; }) ||
      FDT.data.getTrips(corridor.id).find(function (t) { return !t.completed; });
    return [
      "Why is " + peak.id + " risky right now?",
      trip ? "Should " + trip.id + " change its route?" : "Which trips are at risk?",
      "What is the best departure hour for " + peak.id + "?",
      "Any festival traffic today?",
      "How accurate are the models?"
    ];
  }

  // Every answer is built from the ML corridor forecast or the models' real test results (no LLM).
  async function answerFor(question, corridor, params) {
    const text = question.trim();
    const hour = FDT.mlCorridor.hour();
    const hl = pad(hour);
    const forecast = FDT.mlCorridor.get(corridor.id);
    const tripMatch = text.match(/\b(?:TRK|BUS|CAR|VAN)-\d+\b/i);
    const trip = tripMatch && FDT.data.trips.find(function (item) { return item.id.toUpperCase() === tripMatch[0].toUpperCase(); });
    const asked = findSegment(text, params);

    if (asked && asked.corridorId !== corridor.id) {
      return { text: asked.id + " is on the other corridor. Switch the corridor at the top of the screen, or ask about a segment of " + corridor.name + ".", source: "Corridor segment list", href: "#/risk-map", action: "Open Corridor Risk Map" };
    }

    if (trip) {
      const segment = FDT.data.segmentById[trip.currentSegment];
      const status = FDT.state.reroutedTrips.includes(trip.id) ? "Rerouted" : trip.status;
      if (!segment || trip.completed) {
        return { text: trip.id + " is " + status.toLowerCase() + ".", source: trip.id + " trip record (demo fleet)", href: "#/trip/" + encodeURIComponent(trip.id), action: "Open trip" };
      }
      const r = segment.risk[hour];
      const times = bestAndWorst(segment);
      const route = corridor.alternateRoute
        ? "To compare it with the " + corridor.alternateRoute.name + " alternate route, run the Scenario Simulator: it simulates delays from this ML risk. "
        : "This corridor has no alternate route in the prototype, so the main lever is timing. ";
      return {
        text: trip.id + " (" + FDT.data.getVehicle(trip.vehicleId).type + ", " + trip.origin + " → " + trip.destination + ") is on " + segment.id + " " + segment.name + ". The ML risk there at " + hl + " is " + r.toFixed(2) + " (" + band(r) + "), so its status is " + status + ". " + route +
          "Today " + segment.id + " is safest around " + pad(times.best) + " (risk " + segment.risk[times.best].toFixed(2) + ") and riskiest around " + pad(times.worst) + " (" + segment.risk[times.worst].toFixed(2) + ").",
        source: segment.id + " ML forecast for " + forecast.date,
        href: "#/simulator?trip=" + encodeURIComponent(trip.id) + "&segment=" + encodeURIComponent(segment.id),
        action: "Simulate this trip"
      };
    }

    if (/departure|leave|when should|best time|best window|best hour/i.test(text)) {
      const segment = asked || topSegment(corridor, hour);
      const times = bestAndWorst(segment);
      return {
        text: "For " + segment.id + " " + segment.name + ", the ML forecast gives the lowest risk at " + pad(times.best) + " (" + segment.risk[times.best].toFixed(2) + ", " + band(segment.risk[times.best]) + ") and the highest at " + pad(times.worst) + " (" + segment.risk[times.worst].toFixed(2) + "). Right now (" + hl + ") it is " + segment.risk[hour].toFixed(2) + ". " +
          "Most of the hour-to-hour change comes from traffic: peak hours are 08–10 and 17–19. The Departure Planner turns these risks into simulated delay estimates.",
        source: segment.id + " hourly ML risk · " + forecast.date,
        href: "#/departure?segment=" + encodeURIComponent(segment.id),
        action: "Open Departure Planner"
      };
    }

    if (/model|validat|accura|auc|metric|trust|reliab/i.test(text)) {
      const models = await FDT.api.models();
      const rain = models.rainfall.report, acc = models.accident.report, tr = models.traffic.report;
      const persistence = rain && rain.test_baselines["persistence (tomorrow = today)"];
      const peakRule = tr && tr.test_baselines["peak-hour rule (peak=high, else low)"];
      return {
        text: "All three models were tested on the most recent dates they never saw. " +
          (rain ? "Rainfall (" + rain.selected_model + "): log loss " + rain.test_metrics.log_loss.toFixed(3) + " vs " + rain.test_baselines["climatology (city x month frequencies)"].log_loss.toFixed(3) + " for the historical average, but macro-F1 " + rain.test_metrics.macro_f1.toFixed(3) + " is below the simple 'tomorrow = today' rule (" + persistence.macro_f1.toFixed(3) + "); heavy rain stays hard to predict. " : "") +
          (acc ? "Accident (" + acc.selected_model + "): R² " + acc.test_metrics.r2.toFixed(3) + ", because the dataset's risk score is a fixed formula that the model learns exactly, so it is not proof of real-world accuracy. " : "") +
          (tr ? "Traffic (" + tr.selected_model + "): macro-F1 " + tr.test_metrics.macro_f1.toFixed(3) + " vs " + peakRule.macro_f1.toFixed(3) + " for a peak-hour rule. " : "") +
          "The combined corridor risk uses a stated rule, not a trained model, because there are no disruption records.",
        source: "Training reports on the backend",
        href: "#/model",
        action: "View Model & Validation"
      };
    }

    if (/festival|event|holiday|diwali|deepavali|holi|eid|congestion|traffic jam/i.test(text)) {
      const event = forecast.event;
      const busiest = FDT.data.getSegments(corridor.id).slice().sort(function (a, b) { return b.ml.hourly.p_high_congestion[hour] - a.ml.hourly.p_high_congestion[hour]; })[0];
      return {
        text: (event.festival === "None" ? "No festival or event is listed for " + forecast.date + " in significant_dates.csv. " : "Today (" + forecast.date + ") the calendar lists: " + (event.calendar_text || event.festival) + ". The traffic model includes this. ") +
          "The highest chance of heavy congestion at " + hl + " is on " + busiest.id + " " + busiest.name + " (" + pct(busiest.ml.hourly.p_high_congestion[hour]) + ", predicted level " + busiest.ml.hourly.congestion_level[hour] + "). Note: the traffic model learned from traffic density recorded at accident sites, so treat it as an indication.",
        source: "Traffic model + significant_dates.csv",
        href: "#/segment/" + encodeURIComponent(busiest.id),
        action: "Open " + busiest.id
      };
    }

    if (/alert|warning|at risk|which trips/i.test(text)) {
      const alerts = FDT.data.alerts.filter(function (a) { return a.corridorId === corridor.id && a.severity !== "Info" && !FDT.state.closedAlerts.includes(a.id); });
      const atRisk = FDT.data.getTrips(corridor.id).filter(function (t) { return !t.completed && t.status === "At risk"; });
      return {
        text: "There are " + alerts.length + " open ML alerts on " + corridor.name + (alerts.length ? ", for example: " + alerts.slice(0, 2).map(function (a) { return a.title; }).join("; ") + ". " : ". ") +
          (atRisk.length ? atRisk.length + " trip(s) are on a segment with risk ≥ 0.60 right now: " + atRisk.map(function (t) { return t.id; }).join(", ") + "." : "No active trip is on a high-risk segment right now."),
        source: "ML alerts · " + forecast.date,
        href: "#/alerts",
        action: "Open Alerts"
      };
    }

    if (/alternate|nh48a|rerout|route/i.test(text)) {
      return {
        text: corridor.alternateRoute
          ? "The prototype's alternate is " + corridor.alternateRoute.name + " (about " + corridor.alternateRoute.extraKm + " km longer), avoiding the Peth Naka – Kagal stretch (S11–S13). Run the Scenario Simulator to compare both routes: it uses the ML risk of the selected segment and hour to simulate delays."
          : "This corridor has no alternate route defined in the prototype. Use the Departure Planner to pick a lower-risk hour instead.",
        source: "Corridor definition + ML forecast",
        href: "#/simulator",
        action: "Open Scenario Simulator"
      };
    }

    if (asked || /risk|rain|flood|why|segment|forecast|danger|safe/i.test(text)) {
      const segment = asked || topSegment(corridor, hour);
      const why = FDT.mlCorridor.explain(segment, hour);
      const parts = [
        { name: "heavy rain", value: why.rain.value },
        { name: "accident risk", value: why.accident.value },
        { name: "congestion", value: why.congestion.value }
      ].sort(function (a, b) { return b.value - a.value; });
      return {
        text: segment.id + " (" + segment.name + ") has a combined risk of " + why.risk.toFixed(2) + " (" + band(why.risk) + ") at " + hl + " IST. The biggest driver is " + parts[0].name + ". " +
          "Rainfall model: " + pct(why.rain.pHeavy, 1) + " chance of heavy rain (" + FDT.ml.alertRuleShort(why.rain.rule, why.rain.threshold) + (why.rain.alert ? "; alert ON: " + why.rain.reason : "") + "), most likely \"" + why.rain.category.split(" (")[0] + "\", about " + Math.round(why.rain.expectedMm) + " mm expected today. " +
          "Accident model: score " + why.accident.score.toFixed(2) + " for " + why.accident.weather + " weather, " + why.accident.visibility + " visibility and " + why.congestion.level + " traffic. " +
          "Traffic model: " + pct(why.congestion.pHigh) + " chance of high congestion" + (why.event.festival !== "None" ? " (event: " + (why.event.calendar_text || why.event.festival) + ")" : "") + ". " +
          "Inputs: " + why.rain.sourceDetail + (why.rain.city.km === 0 ? "; rainfall measured at this segment's own location." : "; nearest data city " + why.rain.city.name + " (" + why.rain.city.km + " km)."),
        source: segment.id + " ML forecast · " + forecast.date + " " + hl + " IST",
        href: "#/segment/" + encodeURIComponent(segment.id),
        action: "Open segment details"
      };
    }

    return {
      text: "I can explain a segment's risk (\"Why is S05 risky?\"), suggest a departure hour, check a trip (\"Should TRK-2041 change its route?\"), list alerts or festival traffic, and report how accurate the models are. Every answer is built from the ML forecast and shows its source.",
      source: "Assistant capabilities",
      href: "#/guide",
      action: "Read the Prototype Guide"
    };
  }

  // Light, safe formatting for AI answers: escape HTML first, then **bold** and "- " bullets.
  function formatAnswer(text) {
    return FDT.escapeHtml(text).split("\n").map(function (line) {
      const bullet = /^\s*[-*•]\s+/.test(line);
      const html = line.replace(/^\s*[-*•]\s+/, "").replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|\s)#{1,4}\s+/, "$1");
      return bullet ? '<span class="assistant-bullet">• ' + html + "</span>" : html;
    }).join("\n");
  }
  let assistantLabel = "Freight Twin Assistant · built-in answers";

  function addMessage(log, role, content) {
    const message = document.createElement("article");
    message.className = "assistant-message assistant-message--" + role;
    message.setAttribute("aria-label", role === "user" ? "Your question" : "Assistant response");
    const label = document.createElement("span");
    label.className = "assistant-message__label";
    label.textContent = role === "user" ? "You" : assistantLabel;
    const body = document.createElement("p");
    body.className = "assistant-message__body";
    body.textContent = content;
    message.append(label, body);
    log.append(message);
    return message;
  }

  function render(root, params) {
    params = params || {};
    const corridor = FDT.data.getCorridor(FDT.state.corridorId);
    const initialQuery = params.q || "";
    const segment = params.segment && FDT.data.segmentById[params.segment.toUpperCase()];
    const contextLabel = segment && segment.corridorId === corridor.id ? segment.id + " · " + segment.name : corridor.name;
    root.innerHTML =
      '<header class="page-heading"><div><p class="small muted" id="assistant-mode">ML FORECAST · CHECKING AI CONNECTION…</p><h1>GenAI Assistant</h1><p>Ask about corridor risk, trips, reroutes, departure windows, or model validation.</p></div><div class="page-actions"><a class="button" href="#/dashboard">Back to dashboard</a></div></header>' +
      '<div class="assistant-demo-notice" role="note" id="assistant-notice"><strong>Connecting…</strong><span>Checking whether an AI model is configured on the backend.</span></div>' +
      '<div class="assistant-layout"><section class="card assistant-chat-card" aria-label="Assistant conversation"><div class="card__header"><div><h2>Conversation</h2><p>Context: ' + FDT.escapeHtml(contextLabel) + '</p></div><button class="button button--subtle" type="button" id="assistant-clear">Clear chat</button></div>' +
      '<div class="assistant-chat-log" id="assistant-chat-log" role="log" aria-live="polite" aria-relevant="additions text"><article class="assistant-message assistant-message--assistant"><span class="assistant-message__label">Freight Twin Assistant · Demo</span><p class="assistant-message__body">Hi ' + FDT.escapeHtml(FDT.state.user.name.split(/\s+/)[0]) + '. Ask a question about this corridor. Every answer is built from the ML forecast and shows its source.</p><span class="assistant-source">Context: ' + FDT.escapeHtml(contextLabel) + '</span></article></div>' +
      '<form class="assistant-composer" id="assistant-form"><label class="sr-only" for="assistant-question">Ask the assistant a question</label><textarea id="assistant-question" rows="2" maxlength="500" placeholder="Ask about risk, a trip, or a route..." required>' + FDT.escapeHtml(initialQuery) + '</textarea><button type="submit" class="button button--primary">Ask</button></form><p class="assistant-composer-hint">Enter to send · Shift+Enter for a new line · 500 characters max</p></section>' +
      '<aside class="card assistant-prompts"><div class="card__header"><div><h2>Try asking</h2><p>Questions answered from the ML forecast.</p></div></div><div class="assistant-prompt-list">' +
      suggestionsFor(corridor).map(function (suggestion) { return '<button type="button" class="assistant-prompt" data-prompt="' + FDT.escapeHtml(suggestion) + '">' + FDT.escapeHtml(suggestion) + '</button>'; }).join("") +
      '</div><div class="assistant-source-note"><strong>Grounding</strong><p>Each response links to its source. Vehicles and trips are demo data; risk, alerts and metrics come from the ML models.</p></div></aside></div>';

    const log = FDT.$("#assistant-chat-log", root);
    const form = FDT.$("#assistant-form", root);
    const questionField = FDT.$("#assistant-question", root);
    let ai = { enabled: false };
    const history = [];
    function demoTrips() {
      return FDT.data.getTrips(corridor.id).filter(function (t) { return !t.completed; }).map(function (t) {
        return { id: t.id, vehicle: FDT.data.getVehicle(t.vehicleId).type, origin: t.origin, destination: t.destination,
          current_segment: t.currentSegment, status: FDT.state.reroutedTrips.includes(t.id) ? "Rerouted" : t.status };
      });
    }
    function addLinks(response, href, sourceText, actionText) {
      const source = document.createElement("a");
      source.className = "assistant-source";
      source.href = href;
      source.textContent = "Source: " + sourceText;
      response.append(source);
      const action = document.createElement("a");
      action.className = "button button--subtle assistant-response-action";
      action.href = href;
      action.textContent = actionText;
      response.append(action);
    }
    async function submitQuestion(question) {
      const text = question.trim();
      if (!text) {
        questionField.setCustomValidity("Enter a question before sending.");
        questionField.reportValidity();
        return;
      }
      questionField.setCustomValidity("");
      addMessage(log, "user", text);
      questionField.value = "";
      const response = addMessage(log, "assistant", ai.enabled ? "Thinking with " + ai.model + "…" : "Thinking…");
      const body = response.querySelector(".assistant-message__body");
      response.scrollIntoView({ block: "nearest" });
      let aiError = null;
      if (ai.enabled) {
        try {
          const asked = findSegment(text, params);
          const reply = await FDT.api.post("/api/assistant/chat", {
            question: text, corridor_id: corridor.id, date: FDT.state.forecastDate || undefined, hour: FDT.mlCorridor.hour(),
            segment_id: asked ? asked.id : undefined, history: history.slice(-8), trips: demoTrips()
          }, { timeoutMs: 90000 }); // main AI up to 40 s + backup up to 30 s
          history.push({ role: "user", content: text }, { role: "assistant", content: reply.answer });
          body.innerHTML = formatAnswer(reply.answer);
          const answerLabel = response.querySelector(".assistant-message__label");
          if (answerLabel) answerLabel.textContent = "Freight Twin AI · " + reply.model + (reply.fallback_note ? " (backup)" : "");
          if (reply.fallback_note) {
            const note = document.createElement("p");
            note.className = "small muted";
            note.textContent = reply.fallback_note;
            response.append(note);
          }
          const check = document.createElement("p");
          check.className = "assistant-grounding assistant-grounding--" + (reply.grounding.all_supported ? "ok" : "warn");
          check.textContent = reply.grounding.all_supported
            ? "✓ " + (reply.grounding.numbers_checked ? "All " + reply.grounding.numbers_checked + " numbers checked against the ML forecast." : "No numbers to check.")
            : "⚠ Not found in the ML forecast: " + reply.grounding.unsupported.join(", ") + ". Verify these in the source screens before acting.";
          response.append(check);
          addLinks(response, "#/segment/" + encodeURIComponent(reply.focus_segment), "ML forecast " + reply.date + " " + reply.time + " · " + reply.provider + " (" + reply.model + ")", "Open " + reply.focus_segment);
          response.scrollIntoView({ block: "nearest" });
          questionField.focus();
          return;
        } catch (error) {
          if (error.status === 401) return;
          aiError = error;
        }
      }
      let answer;
      try {
        answer = await answerFor(text, corridor, params);
      } catch (error) {
        answer = { text: "I could not build an answer: " + error.message, source: "Backend", href: "#/model", action: "Check backend status" };
      }
      body.textContent = (aiError ? "AI unavailable (" + aiError.message + ") — built-in answer:\n\n" : "") + answer.text;
      addLinks(response, answer.href, answer.source, answer.action);
      response.scrollIntoView({ block: "nearest" });
      questionField.focus();
    }
    // Ask the backend whether a real AI model is configured, and say so clearly.
    FDT.api.get("/api/assistant/status").then(function (status) {
      ai = status;
      const notice = FDT.$("#assistant-notice", root);
      const mode = FDT.$("#assistant-mode", root);
      if (!notice) return;
      if (status.enabled) {
        assistantLabel = "Freight Twin AI · " + status.model;
        mode.textContent = "AI CONNECTED · " + String(status.provider_label).toUpperCase() + " · " + status.model;
        notice.className = "assistant-demo-notice assistant-demo-notice--ai";
        notice.innerHTML = "<strong>AI connected</strong><span>" + FDT.escapeHtml(status.provider_label + " (" + status.model + ")") +
          " answers your questions using the live ML forecast for this corridor, your trips and the models’ test results. Every number in an answer is checked against the forecast." +
          (status.fallback ? " Backup if it is busy: " + FDT.escapeHtml(status.fallback.provider_label + " (" + status.fallback.model + ")") + "." : "") + "</span>";
      } else {
        mode.textContent = "ML FORECAST · BUILT-IN ANSWERS";
        notice.innerHTML = "<strong>AI not connected yet</strong><span>" + FDT.escapeHtml(status.reason || "No AI provider configured.") +
          " Until then, built-in answers from the ML forecast are shown.</span>";
      }
      log.querySelectorAll(".assistant-message--assistant .assistant-message__label").forEach(function (label) { label.textContent = assistantLabel; });
    }).catch(function () { /* keep built-in answers */ });
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      submitQuestion(questionField.value);
    });
    questionField.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
    root.querySelectorAll("[data-prompt]").forEach(function (button) {
      button.addEventListener("click", function () {
        questionField.value = button.getAttribute("data-prompt");
        form.requestSubmit();
      });
    });
    FDT.$("#assistant-clear", root).addEventListener("click", function () {
      history.length = 0;
      log.replaceChildren();
      const welcome = addMessage(log, "assistant", "Chat cleared. Ask a question about the current corridor to get started.");
      const context = document.createElement("span");
      context.className = "assistant-source";
      context.textContent = "Context: " + contextLabel;
      welcome.append(context);
      questionField.focus();
    });
    if (initialQuery) form.requestSubmit();
  }

  FDT.pages.assistant = {
    title: "GenAI Assistant",
    render: render,
    destroy: function () {}
  };
}());
