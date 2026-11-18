import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  BriefcaseBusiness,
  CalendarDays,
  Check,
  ClipboardCheck,
  Code2,
  Compass,
  Search,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import "./Signup.css";

const initialForm = {
  fullName: "",
  email: "",
  password: "",
  confirmPassword: "",
  college: "",
  degree: "B.Tech",
  branch: "",
  academicYear: "1st Year",
  semester: "1st Semester",
  graduationYear: "",
  careerGoal: "AI Engineer",
  codingLevel: "Beginner",
  studyTime: "1 hour",
  programmingLanguage: "Python",
  learningTopics: [],
};

const topics = [
  "Programming Fundamentals",
  "Data Structures & Algorithms",
  "Artificial Intelligence",
  "Machine Learning",
  "Web Development",
  "Database Management",
  "Cloud Computing",
  "System Design",
];

function Signup({ onSignup, onBackToLogin }) {
  const [form, setForm] = useState(initialForm);
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);

  function updateField(event) {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));

    setError("");
  }

  function toggleTopic(topic) {
    setForm((previous) => ({
      ...previous,
      learningTopics: previous.learningTopics.includes(topic)
        ? previous.learningTopics.filter((item) => item !== topic)
        : [...previous.learningTopics, topic],
    }));
  }

  function continueToAcademic(event) {
    event.preventDefault();

    const passwordCategories = [
      /[a-z]/.test(form.password),
      /[A-Z]/.test(form.password),
      /\d/.test(form.password),
      /[^A-Za-z0-9]/.test(form.password),
    ];
    if (form.password.length < 12 || passwordCategories.filter(Boolean).length < 3) {
      setError(
        "Use at least 12 characters and three types: lowercase, uppercase, number, or symbol."
      );
      return;
    }

    if (form.password !== form.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setError("");
    setStep(2);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function continueToPreferences(event) {
    event.preventDefault();

    if (!form.college.trim() || !form.branch.trim()) {
      setError("Please enter your college and branch.");
      return;
    }

    if (!form.graduationYear) {
      setError("Please select your expected graduation year.");
      return;
    }

    setError("");
    setStep(3);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function finishSignup(event) {
    event.preventDefault();

    if (form.learningTopics.length === 0) {
      setError("Please select at least one topic you want to learn.");
      return;
    }

    const account = {
      name: form.fullName.trim(),
      email: form.email.trim(),
      password: form.password,
      profile: {
        college: form.college.trim(),
        degree: form.degree,
        branch: form.branch.trim(),
        academicYear: form.academicYear,
        semester: form.semester,
        graduationYear: Number(form.graduationYear),
        careerGoal: form.careerGoal,
        codingLevel: form.codingLevel,
        studyTime: form.studyTime,
        programmingLanguage: form.programmingLanguage,
        learningTopics: form.learningTopics,
      },
    };

    if (typeof onSignup !== "function") {
      setError("Account creation is unavailable right now. Please try again later.");
      return;
    }

    setLoading(true);
    try {
      await onSignup(account);
    } catch (signupError) {
      setError(signupError.message || "Unable to create your account.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="signup-page">
      <div className="signup-glow signup-glow-one" />
      <div className="signup-glow signup-glow-two" />

      <header className="signup-brand">
        <div className="signup-brand-icon">
        <img src="/quantaai-icon.png" alt="QuantaAI" />
        </div>
        <div>
          <h1>QuantaAI</h1>
          <p>YOUR PERSONAL LEARNING SPACE</p>
        </div>
      </header>

      <section className="signup-layout">
        <div className="signup-intro">
          <span className="signup-pill">
            <Sparkles size={14} aria-hidden="true" /> BUILD YOUR LEARNING JOURNEY
          </span>

          <h2>
            Learn Smarter.
            <br />
            Think Deeper.
            <br />
            <span>Go Further.</span>
          </h2>

          <p className="signup-intro-description">
            Tell us about yourself, your academic journey, and
            the skills you want to build. We'll use these
            preferences to personalize your learning space.
          </p>

          <div className="signup-benefits">
            <div>
              <span>01</span>
              <p>
                <strong>Personal profile</strong>
                <small>Keep your academic information organized.</small>
              </p>
            </div>

            <div>
              <span>02</span>
              <p>
                <strong>Learning preferences</strong>
                <small>Choose your interests and daily study time.</small>
              </p>
            </div>

            <div>
              <span>03</span>
              <p>
                <strong>Career direction</strong>
                <small>Define the skills you want to develop.</small>
              </p>
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

        <div className="signup-card">
          <div className="signup-card-heading">
            <span className="signup-card-symbol"><Sparkles size={18} /></span>

            <h2>Create your account</h2>

            <p>
              Set up your profile in three simple steps.
            </p>
          </div>

          <div className="signup-progress">
            {[1, 2, 3].map((number) => (
              <div
                key={number}
                className={`signup-progress-step ${
                  step >= number ? "active" : ""
                }`}
              >
                <span>{number}</span>
                <small>
                  {number === 1
                    ? "Account"
                    : number === 2
                      ? "Academics"
                      : "Preferences"}
                </small>
              </div>
            ))}
          </div>

          {error && (
            <div className="signup-error" role="alert">
              {error}
            </div>
          )}

          {step === 1 && (
            <form onSubmit={continueToAcademic}>
              <div className="signup-field">
                <label htmlFor="signup-fullName">Full name</label>
                <input
                  id="signup-fullName"
                  name="fullName"
                  value={form.fullName}
                  onChange={updateField}
                  placeholder="Enter your full name"
                  autoComplete="name"
                  required
                />
              </div>

              <div className="signup-field">
                <label htmlFor="signup-email">Email address</label>
                <input
                  id="signup-email"
                  name="email"
                  type="email"
                  value={form.email}
                  onChange={updateField}
                  placeholder="you@example.com"
                  autoComplete="email"
                  required
                />
              </div>

              <div className="signup-field">
                <label htmlFor="signup-password">Password</label>

                <div className="signup-password-wrap">
                  <input
                    id="signup-password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    value={form.password}
                    onChange={updateField}
                    placeholder="At least 12 characters"
                    autoComplete="new-password"
                    minLength={12}
                    required
                  />

                  <button
                    type="button"
                    className="signup-show-password"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? "Hide" : "Show"}
                  </button>
                </div>
              </div>

              <div className="signup-field">
                <label htmlFor="signup-confirmPassword">
                  Confirm password
                </label>

                <input
                  id="signup-confirmPassword"
                  name="confirmPassword"
                  type={showPassword ? "text" : "password"}
                  value={form.confirmPassword}
                  onChange={updateField}
                  placeholder="Enter your password again"
                  autoComplete="new-password"
                  required
                />
              </div>

              <button type="submit" className="signup-primary-button">
                Continue to academics <span>→</span>
              </button>
            </form>
          )}

          {step === 2 && (
            <form onSubmit={continueToPreferences}>
              <div className="signup-field">
                <label htmlFor="signup-college">College / University</label>
                <input
                  id="signup-college"
                  name="college"
                  value={form.college}
                  onChange={updateField}
                  placeholder="Enter your college name"
                  required
                />
              </div>

              <div className="signup-field-row">
                <div className="signup-field">
                  <label htmlFor="signup-degree">Degree</label>
                  <select
                    id="signup-degree"
                    name="degree"
                    value={form.degree}
                    onChange={updateField}
                  >
                    <option>B.Tech</option>
                    <option>B.E.</option>
                    <option>B.Sc.</option>
                    <option>BCA</option>
                    <option>BBA</option>
                    <option>M.Tech</option>
                    <option>MCA</option>
                    <option>Other</option>
                  </select>
                </div>

                <div className="signup-field">
                  <label htmlFor="signup-branch">Branch / Department</label>
                  <input
                    id="signup-branch"
                    name="branch"
                    value={form.branch}
                    onChange={updateField}
                    placeholder="e.g. CSBS"
                    required
                  />
                </div>
              </div>

              <div className="signup-field-row">
                <div className="signup-field">
                  <label htmlFor="signup-academicYear">
                    Current year
                  </label>
                  <select
                    id="signup-academicYear"
                    name="academicYear"
                    value={form.academicYear}
                    onChange={updateField}
                  >
                    <option>1st Year</option>
                    <option>2nd Year</option>
                    <option>3rd Year</option>
                    <option>4th Year</option>
                    <option>Other</option>
                  </select>
                </div>

                <div className="signup-field">
                  <label htmlFor="signup-semester">Semester</label>
                  <select
                    id="signup-semester"
                    name="semester"
                    value={form.semester}
                    onChange={updateField}
                  >
                    <option>1st Semester</option>
                    <option>2nd Semester</option>
                    <option>3rd Semester</option>
                    <option>4th Semester</option>
                    <option>5th Semester</option>
                    <option>6th Semester</option>
                    <option>7th Semester</option>
                    <option>8th Semester</option>
                    <option>Other</option>
                  </select>
                </div>
              </div>

              <div className="signup-field">
                <label htmlFor="signup-graduationYear">
                  Expected graduation year
                </label>
                <select
                  id="signup-graduationYear"
                  name="graduationYear"
                  value={form.graduationYear}
                  onChange={updateField}
                  required
                >
                  <option value="">Select graduation year</option>
                  {Array.from(
                    { length: 9 },
                    (_, index) => new Date().getFullYear() + index
                  ).map((year) => (
                    <option key={year} value={year}>
                      {year}
                    </option>
                  ))}
                </select>
              </div>

              <div className="signup-button-row">
                <button
                  type="button"
                  className="signup-secondary-button"
                  onClick={() => {
                    setError("");
                    setStep(1);
                  }}
                >
                  <ArrowLeft size={15} /> Back
                </button>

                <button type="submit" className="signup-primary-button">
                  Continue <ArrowRight size={15} />
                </button>
              </div>
            </form>
          )}

          {step === 3 && (
            <form onSubmit={finishSignup}>
              <div className="signup-field">
                <label htmlFor="signup-careerGoal">
                  Primary career goal
                </label>
                <select
                  id="signup-careerGoal"
                  name="careerGoal"
                  value={form.careerGoal}
                  onChange={updateField}
                >
                  <option>AI Engineer</option>
                  <option>Software Developer</option>
                  <option>Machine Learning Engineer</option>
                  <option>Data Scientist</option>
                  <option>Web Developer</option>
                  <option>Cybersecurity Engineer</option>
                  <option>Cloud Engineer</option>
                  <option>Researcher</option>
                  <option>Undecided</option>
                </select>
              </div>

              <div className="signup-field">
                <label htmlFor="signup-codingLevel">
                  Current coding level
                </label>
                <select
                  id="signup-codingLevel"
                  name="codingLevel"
                  value={form.codingLevel}
                  onChange={updateField}
                >
                  <option>Beginner</option>
                  <option>Intermediate</option>
                  <option>Advanced</option>
                </select>
              </div>

              <div className="signup-field">
                <label htmlFor="signup-studyTime">
                  Available study time per day
                </label>
                <select
                  id="signup-studyTime"
                  name="studyTime"
                  value={form.studyTime}
                  onChange={updateField}
                >
                  <option>30 minutes</option>
                  <option>1 hour</option>
                  <option>2 hours</option>
                  <option>3+ hours</option>
                </select>
              </div>

              <div className="signup-field">
                <label htmlFor="signup-programmingLanguage">
                  Preferred programming language
                </label>
                <select
                  id="signup-programmingLanguage"
                  name="programmingLanguage"
                  value={form.programmingLanguage}
                  onChange={updateField}
                >
                  <option>Python</option>
                  <option>Java</option>
                  <option>JavaScript</option>
                  <option>C</option>
                  <option>C++</option>
                  <option>Other</option>
                  <option>Not decided yet</option>
                </select>
              </div>

              <fieldset className="signup-topics">
                <legend>What would you like to learn?</legend>

                <p>Select one or more topics.</p>

                <div className="signup-topic-grid">
                  {topics.map((topic) => (
                    <label
                      key={topic}
                      className={`signup-topic-option ${
                        form.learningTopics.includes(topic)
                          ? "selected"
                          : ""
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={form.learningTopics.includes(topic)}
                        onChange={() => toggleTopic(topic)}
                      />

                      <span>{topic}</span>
                    </label>
                  ))}
                </div>
              </fieldset>

              <div className="signup-button-row">
                <button
                  type="button"
                  className="signup-secondary-button"
                  onClick={() => {
                    setError("");
                    setStep(2);
                  }}
                >
                  ← Back
                </button>

                <button
                  type="submit"
                  className="signup-primary-button"
                  disabled={loading}
                >
                  {loading ? "Creating account..." : "Complete profile"}
                  <span>{loading ? "…" : <Check size={15} />}</span>
                </button>
              </div>
            </form>
          )}

          <div className="signup-login-link">
            Already have an account?{" "}
            <button
              type="button"
              onClick={onBackToLogin}
            >
              Back to Login
            </button>
          </div>

          <p className="signup-security-note">
            <ShieldCheck size={13} aria-hidden="true" /> Your account and learning preferences are stored securely.
          </p>
        </div>
      </section>

      <footer className="signup-page-footer">
        <span>© 2026 QuantaAI</span>
        <span>Built for curious minds and ambitious learners.</span>
      </footer>
    </main>
  );
}

export default Signup;