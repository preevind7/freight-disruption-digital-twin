(function () {
  "use strict";
  const FDT = window.FDT = window.FDT || {};
  const roles = [
    { id: "dispatcher", title: "Freight Dispatcher", icon: "truck", description: "Monitor corridor risk and reroute active trips." },
    { id: "operator", title: "Fleet Operator", icon: "fleet", description: "Plan departures and keep vehicles moving." },
    { id: "planner", title: "Highway Planner", icon: "map", description: "Find highway hotspots and plan interventions." },
    { id: "researcher", title: "Researcher / Student", icon: "chart", description: "Explore model performance and validation." }
  ];

  function intro() {
    return [
      '  <div class="login-intro">',
      '    <div class="login-brand"><span class="login-brand__mark" aria-hidden="true">AI</span><span>Freight Disruption Digital Twin</span></div>',
      '    <h1>See disruption before it reaches your fleet.</h1>',
      '    <p class="login-intro__tagline">Forecasting highway disruption in India for every vehicle on the road: trucks, buses, cars/taxis and vans.</p>',
      '    <div class="journey-strip" aria-label="Predict, test, explain, act"><span>Predict</span><b aria-hidden="true">→</b><span>Test</span><b aria-hidden="true">→</b><span>Explain</span><b aria-hidden="true">→</b><span>Act</span></div>',
      '  </div>'
    ].join("");
  }

  // ---------- Sign in ----------
  function signInForm(prefill) {
    return [
      '<form class="login-card" id="login-form" novalidate>',
      '  <h2>Sign in</h2><p>Enter your username and password.</p>',
      prefill && prefill.message ? '  <div class="ml-success" role="status">' + FDT.escapeHtml(prefill.message) + '</div>' : "",
      '  <div class="field"><label for="login-username">Username</label><input id="login-username" name="username" autocomplete="username" required autocapitalize="none" spellcheck="false" value="' + FDT.escapeHtml((prefill && prefill.username) || "") + '"></div>',
      '  <div class="field" style="margin-top:12px"><label for="login-password">Password</label><input id="login-password" name="password" type="password" autocomplete="current-password" required></div>',
      '  <div id="login-error" aria-live="assertive"></div>',
      '  <button class="button button--primary button--wide" type="submit" id="login-submit" style="margin-top:18px">Sign in <span aria-hidden="true">→</span></button>',
      '  <p class="login-switch">New user? <button type="button" class="link-button" data-login-mode="register">Create an account</button></p>',
      '</form>'
    ].join("");
  }

  // ---------- Create account ----------
  function registerForm() {
    return [
      '<form class="login-card" id="register-form" novalidate>',
      '  <h2>Create an account</h2><p>Fill in your details. Your role sets the screens you see first.</p>',
      '  <div class="field"><label for="reg-name">Full name</label><input id="reg-name" autocomplete="name" required minlength="2" maxlength="60" placeholder="e.g. Ravi Kumar"></div>',
      '  <div class="field" style="margin-top:12px"><label for="reg-username">Username</label><input id="reg-username" autocomplete="username" required minlength="3" maxlength="32" pattern="[A-Za-z0-9_.\\-]{3,32}" autocapitalize="none" spellcheck="false" placeholder="e.g. ravi"><small class="field__hint">3–32 characters: letters, numbers, dot, underscore or hyphen. No spaces.</small></div>',
      '  <p class="login-label">Your role</p>',
      '  <div class="role-list" role="radiogroup" aria-label="Your role">',
      roles.map(function (role, index) {
        return '<label class="role-option"><input type="radio" name="role" value="' + role.id + '"' + (index === 0 ? " checked" : "") + '><span class="role-option__icon" aria-hidden="true">' + FDT.icon(role.icon) + '</span><span class="role-option__copy"><strong>' + role.title + '</strong><small>' + role.description + '</small></span></label>';
      }).join(""),
      '  </div>',
      '  <div class="field" style="margin-top:12px"><label for="reg-password">Create password</label><input id="reg-password" type="password" autocomplete="new-password" required minlength="8" maxlength="128"><small class="field__hint">At least 8 characters, with at least one letter and one number.</small></div>',
      '  <div class="field" style="margin-top:12px"><label for="reg-confirm">Confirm password</label><input id="reg-confirm" type="password" autocomplete="new-password" required></div>',
      '  <div id="register-error" aria-live="assertive"></div>',
      '  <button class="button button--primary button--wide" type="submit" id="register-submit" style="margin-top:18px">Create account <span aria-hidden="true">→</span></button>',
      '  <p class="login-switch">Already have an account? <button type="button" class="link-button" data-login-mode="signin">Sign in</button></p>',
      '</form>'
    ].join("");
  }

  function showError(box, error, title) {
    box.innerHTML = FDT.ml.errorBox(error.status === 401 || error.status === 422 ? Object.assign(error, { status: 422 }) : error).replace("Please check the inputs", title);
  }

  function render(root, mode, prefill) {
    root.innerHTML = '<section class="login-page" aria-label="Sign in to Freight Disruption Digital Twin">' + intro() +
      '<div class="login-panel">' + (mode === "register" ? registerForm() : signInForm(prefill)) + '</div></section>';

    root.querySelectorAll("[data-login-mode]").forEach(function (button) {
      button.addEventListener("click", function () { render(root, button.getAttribute("data-login-mode")); });
    });

    if (mode === "register") {
      const form = FDT.$("#register-form", root);
      const password = FDT.$("#reg-password", form);
      const confirm = FDT.$("#reg-confirm", form);
      const errorBox = FDT.$("#register-error", form);
      const submit = FDT.$("#register-submit", form);
      FDT.$("#reg-name", form).focus();
      confirm.addEventListener("input", function () {
        confirm.setCustomValidity(confirm.value && confirm.value !== password.value ? "Passwords do not match." : "");
      });
      form.addEventListener("submit", async function (event) {
        event.preventDefault();
        confirm.setCustomValidity(confirm.value !== password.value ? "Passwords do not match." : "");
        if (!form.reportValidity()) return;
        errorBox.innerHTML = "";
        FDT.ml.loading(submit, true, "Creating account…");
        try {
          const user = await FDT.auth.register({
            name: FDT.$("#reg-name", form).value.trim(),
            username: FDT.$("#reg-username", form).value.trim(),
            password: password.value,
            role: new FormData(form).get("role")
          });
          render(root, "signin", { username: user.username, message: "Account created for " + user.name + ". Sign in with your new password." });
          FDT.$("#login-password", root).focus();
        } catch (error) {
          showError(errorBox, error, "Could not create the account");
        } finally {
          if (submit.isConnected) FDT.ml.loading(submit, false);
        }
      });
      return;
    }

    const form = FDT.$("#login-form", root);
    const usernameField = FDT.$("#login-username", form);
    const passwordField = FDT.$("#login-password", form);
    const errorBox = FDT.$("#login-error", form);
    const submit = FDT.$("#login-submit", form);
    if (!usernameField.value) usernameField.focus();
    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      if (!form.reportValidity()) return;
      errorBox.innerHTML = "";
      FDT.ml.loading(submit, true, "Signing in…");
      try {
        const user = await FDT.auth.login(usernameField.value.trim(), passwordField.value);
        FDT.toast("Signed in as " + user.name + " · " + user.role_title);
        FDT.router.navigate((FDT.roleHomes && FDT.roleHomes[user.role]) || "/dashboard");
      } catch (error) {
        passwordField.value = "";
        showError(errorBox, error, "Sign-in failed");
        passwordField.focus();
      } finally {
        if (submit.isConnected) FDT.ml.loading(submit, false);
      }
    });
  }

  FDT.pages = FDT.pages || {};
  FDT.pages.login = {
    title: "Sign in",
    render: function (root, params) { render(root, params && params.mode === "register" ? "register" : "signin"); },
    destroy: function () {}
  };
}());
