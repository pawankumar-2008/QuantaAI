function Settings({
  theme,
  onThemeChange,
  chatMode,
  onChatModeChange,
  modes,
  backendStatus,
  teacherAvailable,
  quantaLoaded,
  version,
  onBack,
}) {
  const defaultProvider =
    chatMode === "teacher"
      ? "Online"
      : chatMode === "offline"
        ? "QuantaLM"
        : "Automatic · server configured";
  const providerStatus =
    backendStatus !== "Connected"
      ? "Backend unavailable"
      : teacherAvailable
        ? "Available"
        : "Not configured";

  return (
    <div className="settings-page">
      <div className="settings-page-heading">
        <div>
          <span className="eyebrow">PERSONALIZE YOUR WORKSPACE</span>
          <h1>Settings</h1>
          <p>Manage preferences for this browser and your AI workspace.</p>
        </div>
        <button type="button" className="settings-back-button" onClick={onBack}>
          Back to workspace
        </button>
      </div>

      <nav className="settings-navigation" aria-label="Settings sections">
        <a href="#settings-general">General</a>
        <a href="#settings-appearance">Appearance</a>
        <a href="#settings-ai">AI Preferences</a>
        <a href="#settings-about">About</a>
      </nav>

      <div className="settings-sections">
        <section className="settings-card" id="settings-general">
          <div className="settings-section-heading">
            <span className="settings-section-icon" aria-hidden="true">⌘</span>
            <div>
              <h2>General</h2>
              <p>Application preferences for your learning workspace.</p>
            </div>
          </div>
          <div className="settings-row">
            <div>
              <strong>Default assistant</strong>
              <p>The assistant opened with a new workspace.</p>
            </div>
            <span className="settings-value">Surface AI</span>
          </div>
          <div className="settings-row">
            <div>
              <strong>Default AI provider</strong>
              <p>Resolved from your saved interaction mode.</p>
            </div>
            <span className="settings-value">{defaultProvider}</span>
          </div>
        </section>

        <section className="settings-card" id="settings-appearance">
          <div className="settings-section-heading">
            <span className="settings-section-icon" aria-hidden="true">◐</span>
            <div>
              <h2>Appearance</h2>
              <p>Your theme is saved in this browser.</p>
            </div>
          </div>
          <fieldset className="theme-options">
            <legend>Theme preference</legend>
            {["light", "dark"].map((option) => (
              <label className="theme-option" key={option}>
                <input
                  type="radio"
                  name="theme"
                  value={option}
                  checked={theme === option}
                  onChange={() => onThemeChange(option)}
                />
                <span className={`theme-preview theme-preview-${option}`} aria-hidden="true" />
                <span>{option === "light" ? "Light mode" : "Dark mode"}</span>
              </label>
            ))}
          </fieldset>
        </section>

        <section className="settings-card" id="settings-ai">
          <div className="settings-section-heading">
            <span className="settings-section-icon" aria-hidden="true">✦</span>
            <div>
              <h2>AI Preferences</h2>
              <p>Set the mode used for new conversations.</p>
            </div>
          </div>
          <label className="settings-select-row">
            <span>
              <strong>Default interaction mode</strong>
              <small>Saved in this browser and applied to the workspace mode selector.</small>
            </span>
            <select
              value={chatMode}
              onChange={(event) => onChatModeChange(event.target.value)}
              aria-label="Default interaction mode"
            >
              {modes.map((mode) => (
                <option key={mode.value} value={mode.value}>
                  {mode.value === "auto"
                    ? "Automatic"
                    : mode.value === "teacher"
                      ? "Online Mode"
                      : "Offline (QuantaLM)"}
                </option>
              ))}
            </select>
          </label>
          <div className="settings-provider-list" aria-label="Provider availability">
            <div className="settings-provider-row">
              <span>Online</span>
              <strong>{providerStatus}</strong>
            </div>
            <div className="settings-provider-row">
              <span>QuantaLM</span>
              <strong>
                {backendStatus !== "Connected"
                  ? "Backend unavailable"
                  : quantaLoaded
                    ? "Loaded"
                    : "Available · loads on use"}
              </strong>
            </div>
            <p>
              Online Mode and Automatic try OpenRouter Primary, OpenRouter
              Secondary, Google Gemini, then QuantaLM Offline. Offline uses only
              QuantaLM.
            </p>
          </div>
        </section>

        <section className="settings-card" id="settings-about">
          <div className="settings-section-heading">
            <span className="settings-section-icon" aria-hidden="true">ⓘ</span>
            <div>
              <h2>About</h2>
              <p>Application and project information.</p>
            </div>
          </div>
          <div className="settings-row">
            <div>
              <strong>Application</strong>
              <p>Your personal AI learning workspace.</p>
            </div>
            <span className="settings-value">QuantaAI</span>
          </div>
          <div className="settings-row">
            <div>
              <strong>Version</strong>
              <p>Installed frontend version.</p>
            </div>
            <span className="settings-value">{version}</span>
          </div>
          <div className="settings-row">
            <div>
              <strong>Project</strong>
              <p>Multi-agent chat, conversation history, and local QuantaLM support.</p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default Settings;
