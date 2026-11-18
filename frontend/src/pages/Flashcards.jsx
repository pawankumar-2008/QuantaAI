import { useEffect, useRef, useState } from "react";
import {
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  FileText,
  FileUp,
  LoaderCircle,
  RefreshCw,
  RotateCcw,
  X,
} from "lucide-react";
import {
  createFlashcardDeck,
  deleteFlashcardDeck,
  listFlashcardDecks,
  listStudyDocuments,
  uploadStudyDocument,
  updateFlashcardProgress,
} from "../services/api.js";

function renderAnswer(answer) {
  return answer.split(/(\*\*[^*]+\*\*)/g).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    return part;
  });
}

function Flashcards({
  mode,
  onBack,
  onAuthenticationExpired,
  onProgressUpdated,
  onAttemptQuiz,
  initialDeckId,
}) {
  const [topic, setTopic] = useState("");
  const [sourceMode, setSourceMode] = useState("topic");
  const [documentId, setDocumentId] = useState("");
  const [uploadedFilename, setUploadedFilename] = useState("");
  const [documentStatus, setDocumentStatus] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [documentUploading, setDocumentUploading] = useState(false);
  const [cardCount, setCardCount] = useState("8");
  const [difficulty, setDifficulty] = useState("basic");
  const [documents, setDocuments] = useState([]);
  const [decks, setDecks] = useState([]);
  const [deck, setDeck] = useState(null);
  const [revealed, setRevealed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

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
      });
    listFlashcardDecks()
      .then(({ decks: savedDecks }) => {
        if (!cancelled) {
          setDecks(savedDecks);
          setDeck(
            savedDecks.find((item) => item.id === initialDeckId) ||
              savedDecks[0] ||
              null,
          );
        }
      })
      .catch((requestError) => {
        if (cancelled) return;
        if (requestError.status === 401) onAuthenticationExpired();
        else setError(requestError.message || "Unable to load saved flashcards.");
      })
      .finally(() => {
        if (!cancelled) setInitialLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [initialDeckId, onAuthenticationExpired]);

  function changeSourceMode(event) {
    setSourceMode(event.target.value);
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
        setDocuments((current) =>
          current.filter((item) => item.id !== uploaded.id),
        );
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

  async function generateDeck(event) {
    event.preventDefault();
    if (sourceMode === "topic" && !topic.trim()) {
      setError("Enter a topic to generate flashcards.");
      return;
    }
    if (sourceMode !== "topic" && !documentId) {
      setError(
        sourceMode === "upload"
          ? "Upload and index a PDF before generating flashcards."
          : "Select an indexed study document before generating flashcards.",
      );
      return;
    }

    setLoading(true);
    setError("");
    try {
      const request = {
        card_count: Number(cardCount),
        difficulty,
        mode,
      };
      if (sourceMode === "topic") request.topic = topic.trim();
      else request.document_id = Number(documentId);
      const generated = await createFlashcardDeck(request);
      setDeck(generated);
      setRevealed(false);
      setDecks((current) => [generated, ...current]);
      if (sourceMode === "topic") setTopic("");
      await onProgressUpdated();
    } catch (requestError) {
      if (requestError.status === 401) onAuthenticationExpired();
      else setError(requestError.message || "Unable to generate flashcards.");
    } finally {
      setLoading(false);
    }
  }

  async function saveProgress(currentIndex, known) {
    if (!deck) return;
    setLoading(true);
    setError("");
    try {
      const progress = { current_index: currentIndex };
      if (known !== undefined) progress.known = known;
      const updated = await updateFlashcardProgress(deck.id, progress);
      setDeck(updated);
      setDecks((current) =>
        current.map((item) => (item.id === updated.id ? updated : item)),
      );
      setRevealed(false);
      await onProgressUpdated();
    } catch (requestError) {
      if (requestError.status === 401) onAuthenticationExpired();
      else setError(requestError.message || "Unable to save flashcard progress.");
    } finally {
      setLoading(false);
    }
  }

  async function removeDeck(deckId) {
    setLoading(true);
    setError("");
    try {
      await deleteFlashcardDeck(deckId);
      const remaining = decks.filter((item) => item.id !== deckId);
      setDecks(remaining);
      if (deck?.id === deckId) {
        setDeck(remaining[0] || null);
        setRevealed(false);
      }
    } catch (requestError) {
      if (requestError.status === 401) onAuthenticationExpired();
      else setError(requestError.message || "Unable to delete this flashcard deck.");
    } finally {
      setLoading(false);
    }
  }

  const currentIndex = deck?.current_index || 0;
  const currentCard = deck?.cards[currentIndex];
  const knownCards = deck?.known_cards || [];
  const isKnown = knownCards.includes(currentIndex);

  return (
    <section className="learning-page" aria-labelledby="flashcards-heading">
      <header className="learning-page-heading">
        <div>
          <span className="eyebrow">ACTIVE RECALL</span>
          <h1 id="flashcards-heading">Flashcards</h1>
          <p>Study a topic or review key ideas from an uploaded PDF.</p>
        </div>
        <button type="button" className="settings-back-button" onClick={onBack}>
          Back to workspace
        </button>
      </header>

      <form
        className="learning-form flashcard-generation-form"
        onSubmit={generateDeck}
      >
        <input
          ref={fileInputRef}
          className="flashcard-pdf-file-input"
          type="file"
          accept="application/pdf,.pdf"
          onChange={handlePdfUpload}
          aria-label="Choose a PDF for flashcards"
          disabled={documentUploading || loading}
        />
        <label>
          <span>Source</span>
          <select
            value={sourceMode}
            onChange={changeSourceMode}
            aria-label="Flashcard source type"
            disabled={loading || documentUploading}
          >
            <option value="topic">Enter a Topic</option>
            <option value="upload">Upload a PDF</option>
            {documents.length > 0 && (
              <option value="existing">Select an Existing Document</option>
            )}
          </select>
        </label>

        {sourceMode === "topic" && (
          <label>
            <span>Topic</span>
            <input
              value={topic}
              onChange={(event) => setTopic(event.target.value)}
              maxLength={200}
              placeholder="e.g. Binary trees or cell division"
              required
            />
          </label>
        )}

        {sourceMode === "existing" && (
          <div className="flashcard-existing-source">
            <label>
              <span>Existing document</span>
              <select
                value={documentId}
                onChange={(event) => {
                  const selected = documents.find(
                    (item) => String(item.id) === event.target.value,
                  );
                  setDocumentId(event.target.value);
                  setError("");
                  setDocumentStatus(
                    selected ? `${selected.filename} · Indexed and ready` : "",
                  );
                }}
                aria-label="Existing flashcard source document"
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
            {documentId && (
              <p className="flashcard-source-ready" role="status">
                <CheckCircle2 size={13} aria-hidden="true" />
                {documentStatus}
              </p>
            )}
          </div>
        )}

        {sourceMode === "upload" && (
          <div className="flashcard-pdf-source">
            <div className="flashcard-pdf-summary">
              <span className="flashcard-pdf-icon" aria-hidden="true">
                <FileText size={17} />
              </span>
              <span className="flashcard-pdf-copy">
                <strong title={uploadedFilename || undefined}>
                  {uploadedFilename || "No PDF selected"}
                </strong>
                <small>
                  {documentStatus || "The PDF will be indexed before generation."}
                </small>
              </span>
              {documentId && (
                <CheckCircle2
                  className="flashcard-pdf-ready"
                  size={16}
                  aria-label="PDF indexed"
                />
              )}
            </div>
            <div className="flashcard-pdf-actions">
              <button
                type="button"
                className="flashcard-upload-button"
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
                  className="flashcard-remove-pdf"
                  onClick={removeUploadedSelection}
                  disabled={documentUploading || loading}
                  aria-label="Remove selected PDF"
                >
                  <X size={14} aria-hidden="true" />
                  Remove
                </button>
              )}
            </div>
            {documentUploading && (
              <div className="flashcard-upload-progress" role="status">
                <span>
                  {uploadProgress >= 100
                    ? "Upload complete · Processing and indexing PDF…"
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
            aria-label="Flashcard difficulty"
            disabled={loading}
          >
            <option value="basic">Basic</option>
            <option value="medium">Medium</option>
          </select>
        </label>

        <label>
          <span>Cards</span>
          <select
            value={cardCount}
            onChange={(event) => setCardCount(event.target.value)}
            aria-label="Number of flashcards"
          >
            {[3, 5, 8, 12, 20].map((count) => (
              <option key={count} value={count}>{count}</option>
            ))}
          </select>
        </label>
        <button type="submit" disabled={loading || documentUploading}>
          {loading ? `Generating ${difficulty}…` : "Generate flashcards"}
        </button>
      </form>

      {error && <p className="learning-error" role="alert">{error}</p>}

      {initialLoading ? (
        <p className="learning-state" role="status">Loading saved flashcards…</p>
      ) : (
        <div className="flashcard-workspace">
          <section className="flashcard-study" aria-label="Flashcard study area">
            {currentCard ? (
              <>
                <div className="flashcard-study-heading">
                  <div>
                    <span className="eyebrow">{deck.topic}</span>
                    <h2>{deck.title}</h2>
                    <p className="flashcard-deck-summary">
                      {deck.cards.length} cards
                      <span aria-hidden="true">·</span>
                      {deck.difficulty
                        ? `${deck.difficulty} difficulty`
                        : "Difficulty not recorded"}
                    </p>
                  </div>
                  <button
                    type="button"
                    className="flashcard-delete-button"
                    onClick={() => removeDeck(deck.id)}
                    disabled={loading}
                  >
                    Delete deck
                  </button>
                </div>
                <button
                  type="button"
                  className="flashcard-scene"
                  onClick={() => setRevealed((value) => !value)}
                  aria-label={revealed ? "Show flashcard question" : "Show flashcard answer"}
                  aria-pressed={revealed}
                >
                  <span className={`flashcard-flipper ${revealed ? "is-flipped" : ""}`}>
                    <span
                      className="flashcard-face flashcard-front"
                      aria-hidden={revealed}
                    >
                      <span className="flashcard-face-topline">
                        <span className="flashcard-position">
                          {String(currentIndex + 1).padStart(2, "0")} /{" "}
                          {String(deck.cards.length).padStart(2, "0")}
                        </span>
                        {deck.difficulty ? (
                          <span className={`flashcard-difficulty ${deck.difficulty}`}>
                            {deck.difficulty}
                          </span>
                        ) : (
                          <span className="flashcard-difficulty unrecorded">
                            difficulty not recorded
                          </span>
                        )}
                      </span>
                      <span className="flashcard-face-label">QUESTION</span>
                      <span className="flashcard-copy">{currentCard.front}</span>
                      <span className="flashcard-flip-hint">
                        <RotateCcw size={15} aria-hidden="true" />
                        Tap or click to reveal answer
                      </span>
                    </span>
                    <span
                      className="flashcard-face flashcard-back"
                      aria-hidden={!revealed}
                    >
                      <span className="flashcard-face-label">ANSWER</span>
                      <span className="flashcard-copy">
                        {renderAnswer(currentCard.back)}
                      </span>
                      <span className="flashcard-flip-hint">
                        <RotateCcw size={15} aria-hidden="true" />
                        Tap or click to flip back
                      </span>
                    </span>
                  </span>
                </button>
                <div className="flashcard-controls">
                  <button
                    type="button"
                    onClick={() => saveProgress(Math.max(0, currentIndex - 1))}
                    disabled={loading || currentIndex === 0}
                  >
                    <ChevronLeft size={16} aria-hidden="true" />
                    Previous
                  </button>
                  <button
                    type="button"
                    className="flashcard-review-again"
                    onClick={() => saveProgress(currentIndex, false)}
                    disabled={loading}
                  >
                    Not remembered
                  </button>
                  <button
                    type="button"
                    className={`flashcard-known ${isKnown ? "selected" : ""}`}
                    onClick={() => saveProgress(currentIndex, true)}
                    disabled={loading}
                  >
                    <Check size={15} aria-hidden="true" />
                    Remembered
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      saveProgress(Math.min(deck.cards.length - 1, currentIndex + 1))
                    }
                    disabled={loading || currentIndex === deck.cards.length - 1}
                  >
                    Next
                    <ChevronRight size={16} aria-hidden="true" />
                  </button>
                </div>
                <div className="flashcard-progress">
                  <div className="flashcard-progress-labels">
                    <span>Card {currentIndex + 1} of {deck.cards.length}</span>
                    <span>{knownCards.length} remembered</span>
                  </div>
                  <progress
                    value={knownCards.length}
                    max={deck.cards.length}
                    aria-label={`${knownCards.length} of ${deck.cards.length} cards remembered`}
                  />
                </div>
                {currentIndex === deck.cards.length - 1 && (
                  <div className="flashcard-quiz-invitation">
                    <div>
                      <strong>Ready to test your knowledge?</strong>
                      <span>
                        Take a new {deck.difficulty || "basic"} quiz on {deck.topic}.
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => onAttemptQuiz(deck)}
                      disabled={loading}
                    >
                      Attempt Quiz
                      <ChevronRight size={16} aria-hidden="true" />
                    </button>
                  </div>
                )}
              </>
            ) : (
              <div className="learning-empty-state">
                <span aria-hidden="true">▤</span>
                <h2>No flashcard deck selected</h2>
                <p>Generate cards from a topic, or create one from a PDF.</p>
              </div>
            )}
          </section>

          <aside className="flashcard-deck-list" aria-label="Saved flashcard decks">
            <h2>Saved decks</h2>
            {decks.length === 0 ? (
              <p className="learning-empty-inline">Generated decks will appear here.</p>
            ) : (
              <ul>
                {decks.map((item) => (
                  <li key={item.id}>
                    <button
                      type="button"
                      className={deck?.id === item.id ? "selected" : ""}
                      onClick={() => {
                        setDeck(item);
                        setRevealed(false);
                      }}
                      disabled={loading}
                    >
                    <span className="flashcard-deck-item-copy">
                      <strong>{item.title}</strong>
                      <small>{item.topic}</small>
                    </span>
                    <span className="flashcard-deck-item-meta">
                      <small>{item.difficulty || "—"}</small>
                      <small>{item.cards.length} cards</small>
                    </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </aside>
        </div>
      )}
    </section>
  );
}

export default Flashcards;
