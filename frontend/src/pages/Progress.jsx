import { useEffect, useState } from "react";
import { getLearningProgress } from "../services/api.js";

function formatActivity(type) {
  const labels = {
    quiz_created: "Generated a quiz",
    quiz_completed: "Completed a quiz",
    flashcards_created: "Generated flashcards",
    flashcard_reviewed: "Reviewed a flashcard",
  };
  return labels[type] || type.replaceAll("_", " ");
}

function Progress({ onBack, onAuthenticationExpired }) {
  const [progress, setProgress] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    getLearningProgress()
      .then((data) => {
        if (!cancelled) setProgress(data);
      })
      .catch((requestError) => {
        if (cancelled) return;
        if (requestError.status === 401) onAuthenticationExpired();
        else setError(requestError.message || "Unable to load learning progress.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [onAuthenticationExpired]);

  return (
    <section className="learning-page" aria-labelledby="progress-heading">
      <header className="learning-page-heading">
        <div>
          <span className="eyebrow">YOUR LEARNING JOURNEY</span>
          <h1 id="progress-heading">Progress &amp; recommendations</h1>
          <p>Use your quiz results and recent study activity to guide what comes next.</p>
        </div>
        <button type="button" className="settings-back-button" onClick={onBack}>
          Back to workspace
        </button>
      </header>

      {error && <p className="learning-error" role="alert">{error}</p>}
      {loading ? (
        <p className="learning-state" role="status">Loading your learning progress…</p>
      ) : progress ? (
        <>
          <div className="progress-summary-grid">
            <article className="progress-summary-card">
              <span className="eyebrow">COMPLETED QUIZZES</span>
              <strong>{progress.completed_quizzes}</strong>
            </article>
            <article className="progress-summary-card">
              <span className="eyebrow">OVERALL QUIZ SCORE</span>
              <strong>
                {progress.average_score === null
                  ? "—"
                  : `${progress.average_score}%`}
              </strong>
            </article>
            <article className="progress-summary-card">
              <span className="eyebrow">TOPICS TO REVIEW</span>
              <strong>{progress.weak_topics.length}</strong>
            </article>
          </div>

          <section className="learning-history" aria-labelledby="recommendations-heading">
            <div className="learning-section-heading">
              <div>
                <span className="eyebrow">ADAPTIVE REVIEW</span>
                <h2 id="recommendations-heading">Recommendations</h2>
              </div>
            </div>
            {progress.recommendations.length === 0 ? (
              <p className="learning-empty-inline">
                Complete a quiz to get topic-specific review recommendations.
              </p>
            ) : (
              <ul className="recommendation-list">
                {progress.recommendations.map((item) => (
                  <li key={item.topic}>
                    <strong>{item.topic}</strong>
                    <span>{item.message}</span>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <div className="progress-detail-grid">
            <section className="learning-history" aria-labelledby="topic-progress-heading">
              <div className="learning-section-heading">
                <div>
                  <span className="eyebrow">QUIZ PERFORMANCE</span>
                  <h2 id="topic-progress-heading">Topic progress</h2>
                </div>
              </div>
              {progress.topic_progress.length === 0 ? (
                <p className="learning-empty-inline">No topics need review yet.</p>
              ) : (
                <ul className="topic-progress-list">
                  {progress.topic_progress.map((item) => (
                    <li key={item.topic}>
                      <span>
                        <strong>{item.topic}</strong>
                        <small>{item.attempt_count} attempt(s)</small>
                      </span>
                      <strong>{item.average_score}%</strong>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section className="learning-history" aria-labelledby="activity-heading">
              <div className="learning-section-heading">
                <div>
                  <span className="eyebrow">RECENT ACTIVITY</span>
                  <h2 id="activity-heading">Learning activity</h2>
                </div>
              </div>
              {progress.recent_activity.length === 0 ? (
                <p className="learning-empty-inline">Your study activity will appear here.</p>
              ) : (
                <ul className="activity-list">
                  {progress.recent_activity.map((activity, index) => (
                    <li key={`${activity.type}-${activity.created_at}-${index}`}>
                      <span>
                        <strong>{formatActivity(activity.type)}</strong>
                        <small>{activity.topic}</small>
                      </span>
                      <time dateTime={activity.created_at}>
                        {new Date(activity.created_at).toLocaleDateString()}
                      </time>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        </>
      ) : null}
    </section>
  );
}

export default Progress;
