import { useState } from "react";

function Profile({ user, onBack, onUpdate }) {
  const createdAt = user.created_at || user.createdAt;
  const [name, setName] = useState(user.name);
  const [learningGoal, setLearningGoal] = useState(
    user.profile?.learningGoal || user.profile?.careerGoal || "",
  );
  const [learningStyle, setLearningStyle] = useState(
    user.profile?.learningStyle || "",
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);

  async function saveProfile(event) {
    event.preventDefault();
    setSaving(true);
    setError("");
    setSaved(false);
    const profile = { ...(user.profile || {}) };
    if (learningGoal.trim()) profile.learningGoal = learningGoal.trim();
    else delete profile.learningGoal;
    delete profile.careerGoal;
    if (learningStyle.trim()) profile.learningStyle = learningStyle.trim();
    else delete profile.learningStyle;

    try {
      const updatedUser = await onUpdate({ name: name.trim(), profile });
      setName(updatedUser.name);
      setLearningGoal(updatedUser.profile?.learningGoal || "");
      setLearningStyle(updatedUser.profile?.learningStyle || "");
      setSaved(true);
    } catch (saveError) {
      setError(saveError.message || "Unable to update your profile.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="profile-page">
      <div className="settings-page-heading">
        <div>
          <span className="eyebrow">YOUR ACCOUNT</span>
          <h1>My Profile</h1>
          <p>Authenticated account information.</p>
        </div>
        <button type="button" className="settings-back-button" onClick={onBack}>
          Back to workspace
        </button>
      </div>

      <section className="profile-card" aria-label="Profile details">
        <div className="profile-identity">
          <span className="profile-avatar" aria-hidden="true">
            {user.name.trim().charAt(0).toUpperCase()}
          </span>
          <div>
            <h2>{user.name}</h2>
            <p>{user.email}</p>
          </div>
        </div>
        <dl className="profile-details">
          <div>
            <dt>Name</dt>
            <dd>{user.name}</dd>
          </div>
          <div>
            <dt>Registered email</dt>
            <dd>{user.email}</dd>
          </div>
          {user.id !== undefined && user.id !== null && (
            <div>
              <dt>Account ID</dt>
              <dd>{user.id}</dd>
            </div>
          )}
          {createdAt && (
            <div>
              <dt>Account created</dt>
              <dd>
                <time dateTime={createdAt}>
                  {new Intl.DateTimeFormat(undefined, {
                    dateStyle: "long",
                  }).format(new Date(createdAt))}
                </time>
              </dd>
            </div>
          )}
        </dl>
      </section>

      <form className="profile-edit-form" onSubmit={saveProfile}>
        <div className="learning-section-heading">
          <div>
            <span className="eyebrow">PERSONAL DETAILS</span>
            <h2>Edit profile</h2>
          </div>
        </div>
        <label>
          <span>Name</span>
          <input
            value={name}
            onChange={(event) => {
              setName(event.target.value);
              setSaved(false);
            }}
            maxLength={100}
            required
          />
        </label>
        <label>
          <span>Learning goal</span>
          <input
            value={learningGoal}
            onChange={(event) => {
              setLearningGoal(event.target.value);
              setSaved(false);
            }}
            maxLength={200}
            placeholder="What are you working toward?"
          />
        </label>
        <label>
          <span>Preferred learning style</span>
          <input
            value={learningStyle}
            onChange={(event) => {
              setLearningStyle(event.target.value);
              setSaved(false);
            }}
            maxLength={100}
            placeholder="Examples, practice, concise notes…"
          />
        </label>
        {error && <p className="learning-error" role="alert">{error}</p>}
        {saved && <p className="profile-saved-note" role="status">Profile updated.</p>}
        <button type="submit" disabled={saving || !name.trim()}>
          {saving ? "Saving…" : "Save profile"}
        </button>
      </form>
    </div>
  );
}

export default Profile;
