import { useCallback, useEffect, useRef, useState } from "react";
import { FileText, FileUp, LoaderCircle, RefreshCw, X } from "lucide-react";
import {
  createQuiz,
  getQuiz,
  listQuizzes,
  listStudyDocuments,
  submitQuiz,
  uploadStudyDocument,
} from "../services/api.js";

function Quiz({
  mode,
  onBack,
  onAuthenticationExpired,
  onProgressUpdated,
  launchRequest,
  onLaunchHandled,
  onReviewFlashcards,
}) {
  const [topic, setTopic] = useState("");
  const [sourceType, setSourceType] = useState("topic");
  const [documentId, setDocumentId] = useState("");
  const [uploadedFilename, setUploadedFilename] = useState("");
  const [documentStatus, setDocumentStatus] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [documentUploading, setDocumentUploading] = useState(false);
  const [difficulty, setDifficulty] = useState("basic");
  const [questionCount, setQuestionCount] = useState("5");
  const [documents, setDocuments] = useState([]);
  const [documentsLoaded, setDocumentsLoaded] = useState(false);
  const [history, setHistory] = useState([]);
  const [quiz, setQuiz] = useState(null);
  const [answers, setAnswers] = useState({});
  const [result, setResult] = useState(null);
  const [sourceDeckId, setSourceDeckId] = useState(null);
  const [launchNotice, setLaunchNotice] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);
  const handledLaunch = useRef(null);

  useEffect(() => {
    let cancelled = false;
    listStudyDocuments()
      .then((items) => {
        if (!cancelled) setDocuments(items.filter((item) => item.status === "indexed"));
      })
      .catch((requestError) => {
        if (cancelled) return;
        if (requestError.status === 401) onAuthenticationExpired();
        else setError(requestError.message || "Unable to load your study documents.");
      })
      .finally(() => {
        if (!cancelled) setDocumentsLoaded(true);
      });
    listQuizzes()
      .then(({ quizzes }) => {
        if (!cancelled) setHistory(quizzes);
      })
      .catch((requestError) => {
        if (cancelled) return;
        if (requestError.status === 401) onAuthenticationExpired();
        else setError(requestError.message || "Unable to load quiz history.");
      })
      .finally(() => {
        if (!cancelled) setHistoryLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [onAuthenticationExpired]);

  const refreshHistory = useCallback(async () => {
    const { quizzes } = await listQuizzes();
    setHistory(quizzes);
  }, []);

  const generateQuiz = useCallback(async (event, requestOverride) => {
    event?.preventDefault();
    const selectedTopic = requestOverride?.topic ?? topic.trim();
    const selectedDocumentId = requestOverride
      ? requestOverride.document_id || ""
      : documentId;
    const selectedDifficulty = requestOverride?.difficulty ?? difficulty;
    const selectedCount = requestOverride?.question_count ?? Number(questionCount);
    if (!selectedTopic && !selectedDocumentId) {
      setError("Enter a topic or select a study document.");
      return;
    }

    setLoading(true);
    setError("");
    setLaunchNotice(
      requestOverride?.source_unavailable
        ? "The original PDF is no longer available. This quiz will use the deck topic instead."
        : "",
    );
    setQuiz(null);
    setResult(null);
    setAnswers({});
    setSourceDeckId(requestOverride?.deck_id ?? null);
    setDifficulty(selectedDifficulty);
    setTopic(selectedTopic);
    setDocumentId(String(selectedDocumentId));
    setSourceType(selectedDocumentId ? "existing" : "topic");
    try {
      const request = {
        question_count: selectedCount,
        difficulty: selectedDifficulty,
        mode,
      };
      if (selectedTopic) request.topic = selectedTopic;
      if (selectedDocumentId) request.document_id = Number(selectedDocumentId);
      const generated = await createQuiz(request);
      setQuiz(generated);
      await refreshHistory();
      await onProgressUpdated();
    } catch (requestError) {
      if (requestError.status === 401) onAuthenticationExpired();
      else setError(requestError.message || "Unable to generate a quiz.");
    } finally {
      setLoading(false);
    }
  }, [
    difficulty,
    documentId,
    mode,
    onAuthenticationExpired,
    onProgressUpdated,
    questionCount,
    refreshHistory,
    topic,
  ]);

  useEffect(() => {
    if (!launchRequest || !documentsLoaded || handledLaunch.current === launchRequest) {
      return;
    }
    handledLaunch.current = launchRequest;
    const availableDocument = launchRequest.document_id
      ? documents.find((item) => item.id === launchRequest.document_id)
      : null;
    const request = {
      topic: launchRequest.topic,
      difficulty: launchRequest.difficulty || "basic",
      deck_id: launchRequest.deck_id,
    };
    if (availableDocument) {
      request.document_id = availableDocument.id;
    } else if (launchRequest.source_type === "pdf") {
      request.source_unavailable = true;
    }
    generateQuiz(null, request);
    onLaunchHandled();
  }, [launchRequest, documentsLoaded, documents, onLaunchHandled, generateQuiz]);

  async function finishQuiz(event) {
    event.preventDefault();
    if (!quiz) return;
    if (quiz.questions.some((_, index) => answers[index] === undefined)) {
      setError("Answer every question before submitting the quiz.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const submission = await submitQuiz(
        quiz.id,
        quiz.questions.map((_, index) => answers[index]),
      );
      setResult(submission);
      await refreshHistory();
      await onProgressUpdated();
    } catch (requestError) {
      if (requestError.status === 401) onAuthenticationExpired();
      else setError(requestError.message || "Unable to submit your quiz.");
    } finally {
      setLoading(false);
    }
  }

  async function openHistoryQuiz(quizId) {
    setLoading(true);
    setError("");
    setSourceDeckId(null);
    try {
      const savedQuiz = await getQuiz(quizId);
      setQuiz(savedQuiz);
      setAnswers({});
      setResult(
        savedQuiz.completed_at
          ? {
              score: savedQuiz.score,
              question_count: savedQuiz.question_count,
              feedback: savedQuiz.feedback,
              improvement_topics: savedQuiz.improvement_topics,
            }
          : null,
      );
      setDifficulty(savedQuiz.difficulty || "basic");
      setLaunchNotice("");
    } catch (requestError) {
      if (requestError.status === 401) onAuthenticationExpired();
      else setError(requestError.message || "Unable to open this quiz.");
    } finally {
      setLoading(false);
    }
  }

  function tryAnotherQuiz() {
    if (!quiz) return;
    const availableDocument = documents.some(
      (item) => item.id === quiz.document_id,
    )
      ? quiz.document_id
      : null;
    generateQuiz(null, {
      topic: quiz.topic,
      difficulty: quiz.difficulty || difficulty,
      question_count: quiz.question_count,
      document_id: availableDocument,
      deck_id: sourceDeckId,
      source_unavailable: Boolean(quiz.document_id && !availableDocument),
    });
  }

  function changeSourceType(event) {
    setSourceType(event.target.value);
    setTopic("");
    setDocumentId("");
    setUploadedFilename("");
    setDocumentStatus("");
    setUploadProgress(0);
    setError("");
  }

  async function handlePdfUpload(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setSourceType("upload");
    setUploadedFilename(file.name);
    setDocumentId("");
    setDocumentStatus("");
    setUploadProgress(0);
    setError("");
    if (
      !file.name.toLowerCase().endsWith(".pdf") ||
      (file.type && file.type !== "application/pdf")
    ) {
      setDocumentStatus("Select a PDF file to continue.");
      setError("Select a PDF file to upload.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setDocumentStatus("PDF files must be 10 MB or smaller.");
      setError("PDF files must be 10 MB or smaller.");
      return;
    }

    setDocumentUploading(true);
    setDocumentStatus("Uploading PDF…");
    try {
      const uploaded = await uploadStudyDocument(file, setUploadProgress);
      if (uploaded.status === "indexed") {
        setDocuments((current) => [
          uploaded,
          ...current.filter((item) => item.id !== uploaded.id),
        ]);
        setDocumentId(String(uploaded.id));
        setDocumentStatus("PDF indexed and ready");
      } else {
        const statusMessage =
          uploaded.status_message || `${uploaded.filename} could not be indexed.`;
        setDocumentStatus(statusMessage);
        setError(statusMessage);
      }
    } catch (requestError) {
      setDocumentStatus(requestError.message || "Unable to process this PDF.");
      setError(requestError.message || "Unable to upload this PDF.");
      if (requestError.status === 401) onAuthenticationExpired();
    } finally {
      setDocumentUploading(false);
    }
  }

  function removeUploadedSelection() {
    setDocumentId("");
    setUploadedFilename("");
    setDocumentStatus("");
    setUploadProgress(0);
    setError("");
  }

  const correctCount = result?.score ?? 0;
  const totalCount = result?.question_count ?? quiz?.question_count ?? 0;
  const incorrectCount = Math.max(0, totalCount - correctCount);
  const scorePercentage = totalCount
    ? Math.round((correctCount / totalCount) * 100)
    : 0;

  return (
    <section className="learning-page" aria-labelledby="quiz-heading">
      <header className="learning-page-heading">
        <div>
          <span className="eyebrow">PRACTICE &amp; REVIEW</span>
          <h1 id="quiz-heading">Quiz</h1>
          <p>Generate a quiz from a topic or your selected study document.</p>
        </div>
        <button type="button" className="settings-back-button" onClick={onBack}>
          Back to workspace
        </button>
      </header>

      <form className="learning-form quiz-generation-form" onSubmit={generateQuiz}>
        <label>
          <span>Source</span>
          <select
            value={sourceType}
            onChange={changeSourceType}
            aria-label="Quiz source type"
            disabled={loading || documentUploading}
          >
            <option value="topic">Enter a Topic</option>
            <option value="upload">Upload a PDF</option>
            {documents.length > 0 && (
              <option value="existing">Select an Existing Document</option>
            )}
          </select>
        </label>

        {sourceType === "topic" && (
          <label>
            <span>Topic</span>
            <input
              value={topic}
              onChange={(event) => setTopic(event.target.value)}
              maxLength={200}
              placeholder="e.g. Recursion, photosynthesis, or calculus"
              required
            />
          </label>
        )}

        {sourceType === "existing" && (
          <label>
            <span>Existing document</span>
            <select
              value={documentId}
              onChange={(event) => setDocumentId(event.target.value)}
              aria-label="Quiz source document"
              required
              disabled={loading}
            >
              <option value="">Select an indexed document</option>
              {documents.map((document) => (
                <option value={document.id} key={document.id}>
                  {document.filename}
                </option>
              ))}
            </select>
          </label>
        )}

        {sourceType === "upload" && (
          <div className="quiz-pdf-source">
            <input
              ref={fileInputRef}
              className="quiz-pdf-file-input"
              type="file"
              accept="application/pdf,.pdf"
              onChange={handlePdfUpload}
              aria-label="Choose a PDF for quiz generation"
              disabled={documentUploading || loading}
            />
            <div className="quiz-pdf-summary">
              <FileText size={17} aria-hidden="true" />
              <span>
                <strong title={uploadedFilename || undefined}>
                  {uploadedFilename || "No PDF selected"}
                </strong>
                <small>{documentStatus || "The PDF is indexed before generation."}</small>
              </span>
            </div>
            <div className="quiz-pdf-actions">
              <button
                type="button"
                className="quiz-pdf-action"
                onClick={() => fileInputRef.current?.click()}
                disabled={documentUploading || loading}
              >
                {uploadedFilename ? (
                  <RefreshCw size={14} aria-hidden="true" />
                ) : (
                  <FileUp size={14} aria-hidden="true" />
                )}
                {uploadedFilename ? "Replace PDF" : "Choose PDF"}
              </button>
              {uploadedFilename && (
                <button
                  type="button"
                  className="quiz-pdf-action"
                  onClick={removeUploadedSelection}
                  disabled={documentUploading || loading}
                >
                  <X size={14} aria-hidden="true" />
                  Remove
                </button>
              )}
            </div>
            {documentUploading && (
              <div className="quiz-upload-progress" role="status">
                <span>
                  {uploadProgress >= 100
                    ? "Upload complete · Processing PDF…"
                    : uploadProgress > 0
                      ? `Uploading PDF… ${uploadProgress}%`
                      : "Preparing PDF upload…"}
                </span>
                <LoaderCircle size={14} className="flashcard-upload-spinner" />
                <progress
                  max="100"
                  value={uploadProgress || undefined}
                  aria-label="PDF upload progress"
                />
              </div>
            )}
          </div>
        )}

        <label>
          <span>Difficulty</span>
          <select
            value={difficulty}
            onChange={(event) => setDifficulty(event.target.value)}
            aria-label="Quiz difficulty"
            disabled={loading}
          >
            <option value="basic">Basic</option>
            <option value="medium">Medium</option>
          </select>
        </label>
        <label>
          <span>Questions</span>
          <select
            value={questionCount}
            onChange={(event) => setQuestionCount(event.target.value)}
            aria-label="Number of quiz questions"
          >
            {[3, 5, 8, 10].map((count) => (
              <option key={count} value={count}>{count}</option>
            ))}
          </select>
        </label>
        <button type="submit" disabled={loading || documentUploading}>
          {loading && !quiz ? `Generating ${difficulty}…` : "Generate quiz"}
        </button>
      </form>

      {launchNotice && <p className="quiz-launch-notice" role="status">{launchNotice}</p>}
      {error && <p className="learning-error" role="alert">{error}</p>}

      {quiz && (
        <section className="quiz-session" aria-label={`Quiz about ${quiz.topic}`}>
          <div className="quiz-session-heading">
            <div>
              <span className="eyebrow">
                {quiz.provider} · {quiz.model}
              </span>
              <h2>{quiz.topic}</h2>
              <span className={`quiz-difficulty-badge ${quiz.difficulty || difficulty}`}>
                {(quiz.difficulty || difficulty)} difficulty
              </span>
            </div>
            {result && (
              <strong className="quiz-score">{scorePercentage}%</strong>
            )}
          </div>
          {result && (
            <div className="quiz-result-summary" aria-label="Quiz results">
              <div><strong>{totalCount}</strong><span>Total questions</span></div>
              <div><strong>{correctCount}</strong><span>Correct</span></div>
              <div><strong>{incorrectCount}</strong><span>Incorrect</span></div>
              <div><strong>{scorePercentage}%</strong><span>Score</span></div>
            </div>
          )}
          {result?.improvement_topics?.length > 0 && (
            <section className="quiz-improvement-topics" aria-labelledby="quiz-improvements-heading">
              <h3 id="quiz-improvements-heading">Concepts to review</h3>
              <ul>
                {result.improvement_topics.map((item, index) => (
                  <li key={`${quiz.id}-review-${index}`}>{item}</li>
                ))}
              </ul>
            </section>
          )}
          <form onSubmit={finishQuiz}>
            <ol className="quiz-question-list">
              {quiz.questions.map((question, questionIndex) => {
                const feedback = result?.feedback?.[questionIndex];
                return (
                  <li className="quiz-question-card" key={`${quiz.id}-${questionIndex}`}>
                    <h3>{question.question}</h3>
                    <fieldset>
                      <legend>Choose an answer</legend>
                      {question.choices.map((choice, choiceIndex) => {
                        const isCorrect = feedback?.correct_index === choiceIndex;
                        const isSelected = result
                          ? feedback?.selected_index === choiceIndex
                          : answers[questionIndex] === choiceIndex;
                        return (
                          <label
                            className={`quiz-answer ${
                              result && isCorrect ? "correct" : ""
                            } ${result && isSelected && !isCorrect ? "incorrect" : ""}`}
                            key={choiceIndex}
                          >
                            <input
                              type="radio"
                              name={`quiz-${quiz.id}-question-${questionIndex}`}
                              checked={isSelected}
                              disabled={Boolean(result) || loading}
                              onChange={() =>
                                setAnswers((current) => ({
                                  ...current,
                                  [questionIndex]: choiceIndex,
                                }))
                              }
                            />
                            <span>{choice}</span>
                          </label>
                        );
                      })}
                    </fieldset>
                    {feedback && (
                      <p className="quiz-explanation">
                        {feedback.correct ? "Correct. " : "Review. "}
                        {feedback.explanation}
                      </p>
                    )}
                  </li>
                );
              })}
            </ol>
            {!result ? (
              <button type="submit" disabled={loading}>
                {loading ? "Saving results…" : "Submit answers"}
              </button>
            ) : (
              <div className="quiz-result-actions">
                <button
                  type="button"
                  onClick={() => onReviewFlashcards(sourceDeckId)}
                  disabled={loading}
                >
                  Review Flashcards Again
                </button>
                <button
                  type="button"
                  onClick={tryAnotherQuiz}
                  disabled={loading}
                >
                  Try Another Quiz
                </button>
              </div>
            )}
          </form>
        </section>
      )}

      <section className="learning-history" aria-labelledby="quiz-history-heading">
        <div className="learning-section-heading">
          <div>
            <span className="eyebrow">YOUR PRACTICE</span>
            <h2 id="quiz-history-heading">Quiz history</h2>
          </div>
        </div>
        {historyLoading ? (
          <p className="learning-state" role="status">Loading quiz history…</p>
        ) : history.length === 0 ? (
          <p className="learning-empty-inline">Completed and active quizzes will appear here.</p>
        ) : (
          <ul className="learning-history-list">
            {history.map((item) => (
              <li key={item.id}>
                <span>
                  <strong>{item.topic}</strong>
                  <small>
                    {item.difficulty || "difficulty not recorded"} ·{" "}
                    {item.score === null
                      ? "Not submitted"
                      : `${item.score}/${item.question_count} correct`}
                  </small>
                </span>
                <button
                  type="button"
                  onClick={() => openHistoryQuiz(item.id)}
                  disabled={loading}
                >
                  Review
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </section>
  );
}

export default Quiz;
