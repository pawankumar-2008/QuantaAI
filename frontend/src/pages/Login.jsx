import { useState } from "react";
import {
  ArrowRight,
  BookOpen,
  Brain,
  BriefcaseBusiness,
  CalendarDays,
  ClipboardCheck,
  Code2,
  Compass,
  Eye,
  EyeOff,
  LockKeyhole,
  Mail,
  Search,
  Target,
} from "lucide-react";
import "./Login.css";

function Login({ onLogin, onSignup }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");

    if (!email.trim() || !password) {
      setError("Please enter your email and password.");
      return;
    }

    if (!email.includes("@") || !email.includes(".")) {
      setError("Please enter a valid email address.");
      return;
    }

    if (typeof onLogin !== "function") {
      setError("Sign in is unavailable right now. Please try again later.");
      return;
    }

    setLoading(true);
    try {
      await onLogin({ email: email.trim(), password });
    } catch (loginError) {
      setError(loginError.message || "Unable to sign in. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function handleSignup() {
    if (typeof onSignup === "function") {
      onSignup();
    }
  }

  return (
    <main className="login-page">
      <div className="login-background-glow login-glow-one" />
      <div className="login-background-glow login-glow-two" />

      <section className="login-layout">
        <div className="login-showcase">
          <header className="login-brand">
            <div className="login-brand-icon">
              <img
                src="/quantaai-icon.png"
                alt="QuantaAI logo"
                onError={(event) => {
                  event.currentTarget.style.display = "none";
                }}
              />
            </div>

            <div className="login-brand-text">
              <h1>QuantaAI</h1>
              <p>YOUR PERSONAL LEARNING SPACE</p>
            </div>
          </header>

          <div className="login-hero">
            <div className="login-eyebrow">
              <span className="login-status-dot" />
              YOUR LEARNING JOURNEY STARTS HERE
            </div>

            <h2>
              Learn Smarter.
              <br />
              Think Deeper.
              <br />
              <span>Go Further.</span>
            </h2>

            <p className="login-hero-description">
              One intelligent learning space for your studies, coding
              practice, career preparation, and daily goals.
            </p>

            <div className="login-feature-list">
              <div className="login-feature">
                <span className="login-feature-icon"><Brain size={18} /></span>
                <div>
                  <h3>Personalized learning</h3>
                  <p>Learn concepts at your own pace.</p>
                </div>
              </div>

              <div className="login-feature">
                <span className="login-feature-icon"><Compass size={18} /></span>
                <div>
                  <h3>Seven dedicated AI agents</h3>
                  <p>Study, code, plan, explore, and prepare.</p>
                </div>
              </div>

              <div className="login-feature">
                <span className="login-feature-icon"><Target size={18} /></span>
                <div>
                  <h3>Progress with purpose</h3>
                  <p>Organize your learning around your goals.</p>
                </div>
              </div>
            </div>

            <div className="auth-agent-roster" aria-label="Seven specialist agents">
              <span><BookOpen size={13} /> Study</span>
              <span><Code2 size={13} /> Coding</span>
              <span><BriefcaseBusiness size={13} /> Career</span>
              <span><Compass size={13} /> Guide</span>
              <span><CalendarDays size={13} /> Planner</span>
              <span><ClipboardCheck size={13} /> Exam</span>
              <span><Search size={13} /> Research</span>
            </div>
          </div>

          <div className="login-showcase-footer">
            <span>LEARN</span>
            <span className="login-footer-separator">•</span>
            <span>CREATE</span>
            <span className="login-footer-separator">•</span>
            <span>GROW</span>
          </div>
        </div>

        <div className="login-card-wrapper">
          <div className="login-card">
            <div className="login-card-heading">
              <div className="login-mobile-logo">
                <img
                  src="/quantaai-icon.png"
                  alt="QuantaAI logo"
                  onError={(event) => {
                    event.currentTarget.style.display = "none";
                  }}
                />
              </div>

              <span className="login-welcome-label">WELCOME BACK</span>

              <h2>Sign in to QuantaAI</h2>

              <p>
                Continue your learning journey and pick up where you left
                off.
              </p>
            </div>

            <form className="login-form" onSubmit={handleSubmit}>
              <div className="login-field">
                <label htmlFor="login-email">Email address</label>

                <div className="login-input-wrapper">
                  <span className="login-input-icon" aria-hidden="true">
                    <Mail size={15} />
                  </span>

                  <input
                    id="login-email"
                    name="email"
                    type="email"
                    placeholder="you@example.com"
                    autoComplete="email"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    required
                  />
                </div>
              </div>

              <div className="login-field">
                <div className="login-password-heading">
                  <label htmlFor="login-password">Password</label>

                  <button
                    className="login-forgot-button"
                    type="button"
                    onClick={() =>
                      setError(
                        "Password recovery will be available after account authentication is connected."
                      )
                    }
                  >
                    Forgot password?
                  </button>
                </div>

                <div className="login-input-wrapper">
                  <span className="login-input-icon" aria-hidden="true">
                    <LockKeyhole size={15} />
                  </span>

                  <input
                    id="login-password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Enter your password"
                    autoComplete="current-password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    required
                  />

                  <button
                    className="login-password-toggle"
                    type="button"
                    onClick={() => setShowPassword((previous) => !previous)}
                    aria-label={
                      showPassword ? "Hide password" : "Show password"
                    }
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              {error && (
                <div className="login-error" role="alert">
                  {error}
                </div>
              )}

              <button
                className="login-submit-button"
                type="submit"
                disabled={loading}
              >
                <span>{loading ? "Signing in..." : "Sign in"}</span>
                <ArrowRight size={17} aria-hidden="true" />
              </button>
            </form>

            <div className="login-divider">
              <span>NEW TO QUANTAAI?</span>
            </div>

            <button
              className="login-create-account-button"
              type="button"
              onClick={handleSignup}
            >
              Create an account
              <ArrowRight size={16} aria-hidden="true" />
            </button>

            <p className="login-security-note">
              Your personal learning space, designed around you.
            </p>
          </div>

          <p className="login-bottom-note">
            QUANTAAI <span>•</span> YOUR NEXT CHAPTER STARTS HERE
          </p>
        </div>
      </section>
    </main>
  );
}

export default Login;