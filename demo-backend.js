/* Demo backend: lets this frontend run on GitHub Pages with no server.
   Every request the app would send to the FastAPI backend is answered here, in the browser,
   from a saved sample day (js/demo-data.js). Nothing is sent anywhere.
   - Corridor risk, alerts and model metrics: saved from the real backend for 19 Aug 2025.
   - Live predictions (Predict pages) and the real AI assistant need the full app, so they say so. */
(function () {
  "use strict";
  const FDT = window.FDT = window.FDT || {};
  const DATA = window.FDT_DEMO_DATA;
  if (!DATA) return;
  // Demo mode only on GitHub Pages (or with ?demo in the address). On the laptop, the tunnel link or
  // http://127.0.0.1:8000 the app talks to the real backend as usual. A backend address set with
  // localStorage "fdt-api-base" or window.FDT_API_BASE also turns the demo off.
  let customBackend = null;
  try { customBackend = window.localStorage.getItem("fdt-api-base"); } catch (error) { customBackend = null; }
  const askedForDemo = /(^|[?&])demo(=|&|$)/.test(location.search.slice(1));
  if (window.FDT_API_BASE || customBackend || !(/github\.io$/.test(location.hostname) || askedForDemo)) return;

  const ROLES = {
    dispatcher: "Freight Dispatcher",
    operator: "Fleet Operator",
    planner: "Highway Planner",
    researcher: "Researcher"
  };
  const NEEDS_FULL_APP = "Demo mode: this needs the full app with the backend (live ML predictions and the real AI assistant). " +
    "The demo shows a saved sample day from the ML models (19 Aug 2025).";
  const ACCOUNTS_KEY = "fdt-demo-accounts";

  function copy(value) { return JSON.parse(JSON.stringify(value)); }
  function fail(message, status) {
    const error = new Error(message);
    error.status = status;
    error.errors = [];
    return error;
  }
  function accounts() {
    try { return JSON.parse(window.localStorage.getItem(ACCOUNTS_KEY) || "{}"); } catch (error) { return {}; }
  }
  function saveAccount(account) {
    try {
      const all = accounts();
      all[account.username.toLowerCase()] = account;
      window.localStorage.setItem(ACCOUNTS_KEY, JSON.stringify(all));
    } catch (error) { /* storage blocked: the account works for this visit only */ }
  }
  function userFor(username) {
    const key = String(username || "").trim().toLowerCase();
    const saved = accounts()[key];
    if (saved) return { username: saved.username, role: saved.role, role_title: ROLES[saved.role], name: saved.name };
    const role = ROLES[key] ? key : "dispatcher";
    return { username: key || "demo", role: role, role_title: ROLES[role], name: "Demo " + ROLES[role] };
  }
  function params(path) {
    const query = path.split("?")[1] || "";
    const out = {};
    query.split("&").filter(Boolean).forEach(function (pair) {
      const parts = pair.split("=");
      out[decodeURIComponent(parts[0])] = decodeURIComponent((parts[1] || "").replace(/\+/g, " "));
    });
    return out;
  }

  function answer(path, options) {
    const route = path.split("?")[0];
    const body = options.body || {};
    if (route === "/api/auth/login") {
      if (!body.username || !body.password) throw fail("Enter a username and password.", 422);
      return {
        access_token: "demo-token",
        expires_at: new Date(Date.now() + 7 * 864e5).toISOString(),
        user: userFor(body.username)
      };
    }
    if (route === "/api/auth/register") {
      const role = ROLES[body.role] ? body.role : "dispatcher";
      const account = { username: String(body.username || "").trim(), name: String(body.name || "").trim() || "Demo user", role: role };
      if (!account.username) throw fail("Choose a username.", 422);
      saveAccount(account);
      return { username: account.username, name: account.name, role: role, role_title: ROLES[role] };
    }
    if (route === "/api/auth/me") {
      const user = FDT.state && FDT.state.user;
      return user ? { username: user.username, role: user.role, role_title: user.roleTitle, name: user.name } : userFor("dispatcher");
    }
    if (route === "/api/corridor/forecast") {
      const id = params(path).corridor_id || "mumbai-bengaluru";
      const forecast = DATA.forecasts[id];
      if (!forecast) throw fail("Unknown corridor in the demo.", 404);
      return copy(forecast);
    }
    if (route === "/api/models") return copy(DATA.models);
    if (route === "/api/assistant/status") {
      return { enabled: false, provider: "none", provider_label: null, model: null, fallback: null,
        reason: "Demo mode: the real AI (Gemini with Groq backup) runs only in the full app. Built-in answers below use the saved sample day." };
    }
    throw fail(NEEDS_FULL_APP, 503);
  }

  FDT.demo = {
    enabled: true,
    date: DATA.date,
    handle: function (path, options) {
      return new Promise(function (resolve, reject) {
        // A short pause so loading states look natural.
        window.setTimeout(function () {
          try { resolve(answer(path, options || {})); } catch (error) { reject(error); }
        }, 120);
      });
    }
  };
}());
