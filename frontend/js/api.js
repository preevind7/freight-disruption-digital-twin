(function () {
  "use strict";
  const FDT = window.FDT = window.FDT || {};

  // Backend address. Override without editing code:
  //   localStorage.setItem("fdt-api-base", "http://127.0.0.1:8000")
  // Opened from the backend itself (e.g. a public trycloudflare.com link or http://127.0.0.1:8000/) -> same address.
  // Opened from the separate frontend server (port 5500), GitHub Pages or a file -> local backend on port 8000.
  const SEPARATE_FRONTEND = location.protocol === "file:" || location.port === "5500" || /github\.io$/.test(location.hostname);
  const DEFAULT_API_BASE = SEPARATE_FRONTEND ? "http://127.0.0.1:8000" : location.origin;
  const TIMEOUT_MS = 15000;

  function baseUrl() {
    let saved = null;
    try { saved = window.localStorage.getItem("fdt-api-base"); } catch (error) { saved = null; }
    return (saved || window.FDT_API_BASE || DEFAULT_API_BASE).replace(/\/+$/, "");
  }

  function ApiError(message, status, errors) {
    const error = new Error(message);
    error.status = status;
    error.errors = errors || [];
    return error;
  }

  async function request(path, options) {
    options = options || {};
    // Demo mode (GitHub Pages only): answer from the saved sample in the browser, no server needed.
    if (FDT.demo && FDT.demo.enabled) return FDT.demo.handle(path, options);
    const controller = new AbortController();
    const timer = window.setTimeout(function () { controller.abort(); }, options.timeoutMs || TIMEOUT_MS);
    let response;
    try {
      const headers = {};
      if (options.body) headers["Content-Type"] = "application/json";
      if (FDT.state && FDT.state.token && !options.skipAuth) headers.Authorization = "Bearer " + FDT.state.token;
      response = await fetch(baseUrl() + path, {
        method: options.method || "GET",
        headers: headers,
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: controller.signal
      });
    } catch (error) {
      if (error.name === "AbortError") {
        throw ApiError("No answer within " + Math.round((options.timeoutMs || TIMEOUT_MS) / 1000) + " seconds. The backend is running but is still busy; please try again.", 0);
      }
      throw ApiError("Cannot reach the backend at " + baseUrl() + ". Start it with: uvicorn app.main:app --reload (in the backend folder).", 0);
    } finally {
      window.clearTimeout(timer);
    }
    let data = null;
    try { data = await response.json(); } catch (error) { data = null; }
    if (response.status === 401 && !options.skipAuth) {
      // Token missing, expired or revoked: sign out and send the user to the login page.
      FDT.auth.endSession(data && data.detail ? data.detail : "Your session has ended. Please sign in again.");
      throw ApiError(data && data.detail ? data.detail : "Please sign in again.", 401);
    }
    if (!response.ok) {
      const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : "Request failed with status " + response.status + ".";
      throw ApiError(detail, response.status, data && data.errors);
    }
    return data;
  }

  let modelsPromise = null;

  FDT.auth = {
    // POST /api/auth/login -> save token + profile in the app state.
    login: async function (username, password) {
      const data = await request("/api/auth/login", { method: "POST", body: { username: username, password: password }, skipAuth: true });
      modelsPromise = null;
      FDT.store.update(function (state) {
        state.token = data.access_token;
        state.tokenExpiresAt = data.expires_at;
        state.user = { username: data.user.username, role: data.user.role, roleTitle: data.user.role_title, name: data.user.name };
      });
      return data.user;
    },
    // POST /api/auth/register -> creates the account (does not sign in).
    register: function (details) {
      return request("/api/auth/register", { method: "POST", body: details, skipAuth: true });
    },
    endSession: function (message) {
      modelsPromise = null;
      if (!FDT.state.token && !FDT.state.user) return;
      FDT.store.update(function (state) { state.user = null; state.token = null; state.tokenExpiresAt = null; });
      if (typeof FDT.toast === "function" && message) FDT.toast(message);
      if (FDT.router) FDT.router.navigate("/login");
    }
  };

  FDT.api = {
    baseUrl: baseUrl,
    get: function (path) { return request(path); },
    post: function (path, body, options) { return request(path, Object.assign({ method: "POST", body: body }, options || {})); },
    // Model metadata (allowed input values, metrics). Cached; cleared on failure so Retry works.
    models: function (refresh) {
      if (refresh || !modelsPromise) {
        modelsPromise = request("/api/models").catch(function (error) { modelsPromise = null; throw error; });
      }
      return modelsPromise;
    }
  };

  // ---------- small UI helpers shared by the ML pages ----------
  const esc = function (value) { return FDT.escapeHtml(value === null || value === undefined ? "" : value); };
  FDT.ml = {
    esc: esc,
    percent: function (value, digits) { return (Number(value) * 100).toFixed(digits === undefined ? 1 : digits) + "%"; },
    // Heavy-rain alert rule chosen on validation data by the backend training (model / persistence / both).
    alertRuleShort: function (rule, threshold) {
      const pct = (Number(threshold) * 100).toFixed(1) + "%";
      if (rule === "persistence") return "alert if heavy rain the day before";
      if (rule === "forecast") return "alert if 64.5 mm or more is forecast";
      if (rule === "model_or_persistence") return "alert at " + pct + " or heavy rain the day before";
      return "alert at " + pct;
    },
    levelKind: function (level) {
      const key = String(level).toLowerCase();
      return key === "high" ? "high" : key === "moderate" || key === "medium" ? "moderate" : "low";
    },
    options: function (values, selected, labels) {
      return values.map(function (value) {
        const label = labels && labels[value] ? labels[value] : value;
        return '<option value="' + esc(value) + '"' + (String(value) === String(selected) ? " selected" : "") + ">" + esc(label) + "</option>";
      }).join("");
    },
    bars: function (items, options) {
      options = options || {};
      const max = options.max || Math.max(0.0001, ...items.map(function (item) { return Math.abs(item.value); }));
      return '<div class="ml-bars">' + items.map(function (item) {
        const width = Math.max(1, Math.abs(item.value) / max * 100);
        const kind = item.kind || (item.value < 0 ? "down" : "up");
        return '<div class="ml-bar"><span class="ml-bar__label">' + esc(item.label) + '</span><span class="ml-bar__track"><span class="ml-bar__fill ml-bar__fill--' + kind + '" style="width:' + width.toFixed(1) + '%"></span></span><span class="ml-bar__value number">' + esc(item.text) + "</span></div>";
      }).join("") + "</div>";
    },
    loading: function (button, isLoading, label) {
      if (isLoading) {
        button.dataset.label = button.innerHTML;
        button.disabled = true;
        button.innerHTML = '<span class="loading-spinner" aria-hidden="true"></span> ' + esc(label || "Predicting…");
      } else {
        button.disabled = false;
        if (button.dataset.label) button.innerHTML = button.dataset.label;
      }
    },
    errorBox: function (error) {
      const items = (error.errors || []).filter(function (item) { return item.field; });
      return '<div class="ml-error" role="alert"><strong>' + (error.status === 503 ? "Model not available" : error.status === 0 ? "Backend unreachable" : error.status === 422 ? "Please check the inputs" : "Request failed") + '</strong><p>' + esc(error.message) + "</p>" +
        (items.length > 1 ? "<ul>" + items.map(function (item) { return "<li><code>" + esc(item.field) + "</code>: " + esc(item.message) + "</li>"; }).join("") + "</ul>" : "") + "</div>";
    },
    // Full-page state when model metadata cannot be loaded.
    unavailable: function (root, title, error, retry) {
      root.innerHTML = '<header class="page-heading"><div><p class="small muted">ML PREDICTIONS · ' + (FDT.demo ? "Demo" : "Live backend") + '</p><h1>' + esc(title) + "</h1></div></header>" + FDT.ml.errorBox(error) +
        '<p><button type="button" class="button button--primary" data-ml-retry>Retry</button></p>';
      root.querySelector("[data-ml-retry]").addEventListener("click", retry);
    },
    skeleton: function (root, title) {
      root.innerHTML = '<header class="page-heading"><div><p class="small muted">ML PREDICTIONS · ' + (FDT.demo ? "Demo" : "Live backend") + '</p><h1>' + esc(title) + '</h1><p>Loading model details from the backend…</p></div></header><div class="card"><div class="skeleton" style="width:40%;margin-bottom:14px"></div><div class="skeleton" style="width:75%;margin-bottom:10px"></div><div class="skeleton" style="width:55%"></div></div>';
    },
    modelLine: function (model) {
      return '<p class="ml-model-line small muted">Model: <strong>' + esc(model.algorithm) + "</strong> · trained " + esc(String(model.trained_at || "").replace("T", " ").replace("+00:00", " UTC")) + "</p>";
    }
  };

  // When the app opens with a saved token, confirm with the backend that it is still valid
  // (e.g. after the server secret changed or the account was removed).
  document.addEventListener("DOMContentLoaded", function () {
    window.setTimeout(function () {
      if (FDT.state && FDT.state.token) FDT.api.get("/api/auth/me").catch(function () { /* 401 handled above; offline ignored */ });
    }, 0);
  });
}());
