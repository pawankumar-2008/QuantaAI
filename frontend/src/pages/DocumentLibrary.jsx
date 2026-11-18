import { useEffect, useMemo, useState } from "react";
import {
  ArrowDownWideNarrow,
  Download,
  Eye,
  FileText,
  LoaderCircle,
  Pencil,
  Search,
  Trash2,
} from "lucide-react";
import {
  deleteStudyDocument,
  getStudyDocumentFile,
  listStudyDocuments,
  renameStudyDocument,
} from "../services/api.js";

function formatFileSize(sizeBytes) {
  if (sizeBytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(sizeBytes / 1024))} KB`;
  }
  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatUploadDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Upload date unavailable"
    : date.toLocaleString();
}

function DocumentLibrary({ onBack, onAuthenticationExpired, onDocumentsChanged }) {
  const [documents, setDocuments] = useState([]);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortOrder, setSortOrder] = useState("newest");
  const [editingId, setEditingId] = useState(null);
  const [nameDraft, setNameDraft] = useState("");
  const [confirmingDeleteId, setConfirmingDeleteId] = useState(null);
  const [mutatingId, setMutatingId] = useState(null);
  const [fileAction, setFileAction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    listStudyDocuments()
      .then((items) => {
        if (cancelled) return;
        setDocuments(items);
        onDocumentsChanged(items);
      })
      .catch((requestError) => {
        if (cancelled) return;
        if (requestError.status === 401) onAuthenticationExpired();
        else setError(requestError.message || "Unable to load your documents.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [onAuthenticationExpired, onDocumentsChanged]);

  const visibleDocuments = useMemo(() => {
    const normalizedSearch = search.trim().toLocaleLowerCase();
    return [...documents]
      .filter((document) =>
        normalizedSearch
          ? document.filename.toLocaleLowerCase().includes(normalizedSearch)
          : true,
      )
      .filter((document) =>
        statusFilter === "all" ? true : document.status === statusFilter,
      )
      .sort((left, right) => {
        const dateDifference =
          new Date(right.created_at).getTime() -
          new Date(left.created_at).getTime();
        const ordered = Number.isFinite(dateDifference) ? dateDifference : 0;
        return (sortOrder === "newest" ? ordered : -ordered) ||
          right.id - left.id;
      });
  }, [documents, search, sortOrder, statusFilter]);

  async function openDocument(document) {
    setError("");
    const viewer = window.open("about:blank", "_blank");
    if (!viewer) {
      setError("Allow pop-ups for QuantaAI to view this PDF in a new tab.");
      return;
    }

    viewer.opener = null;
    viewer.document.title = "Opening PDF…";
    viewer.document.body.textContent = "Loading PDF…";
    setFileAction({ id: document.id, type: "view" });
    try {
      const file = await getStudyDocumentFile(document.id);
      const url = URL.createObjectURL(file);
      viewer.location.replace(url);
      const releaseWhenClosed = window.setInterval(() => {
        if (viewer.closed) {
          window.clearInterval(releaseWhenClosed);
          URL.revokeObjectURL(url);
        }
      }, 1000);
    } catch (requestError) {
      viewer.close();
      if (requestError.status === 401) onAuthenticationExpired();
      else setError(requestError.message || "Unable to open this PDF.");
    } finally {
      setFileAction(null);
    }
  }

  async function downloadDocument(document) {
    setError("");
    setFileAction({ id: document.id, type: "download" });
    try {
      const file = await getStudyDocumentFile(document.id, { download: true });
      const url = URL.createObjectURL(file);
      const link = window.document.createElement("a");
      link.href = url;
      link.download = document.filename;
      link.hidden = true;
      window.document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
    } catch (requestError) {
      if (requestError.status === 401) onAuthenticationExpired();
      else setError(requestError.message || "Unable to download this PDF.");
    } finally {
      setFileAction(null);
    }
  }

  async function saveName(document) {
    const filename = nameDraft.trim();
    if (!filename) {
      setError("Document name cannot be empty.");
      return;
    }
    setMutatingId(document.id);
    setError("");
    try {
      const updated = await renameStudyDocument(document.id, filename);
      const nextDocuments = documents.map((item) =>
        item.id === updated.id ? updated : item,
      );
      setDocuments(nextDocuments);
      onDocumentsChanged(nextDocuments);
      setEditingId(null);
      setNameDraft("");
    } catch (requestError) {
      if (requestError.status === 401) onAuthenticationExpired();
      else setError(requestError.message || "Unable to rename this document.");
    } finally {
      setMutatingId(null);
    }
  }

  async function removeDocument(document) {
    setMutatingId(document.id);
    setError("");
    try {
      await deleteStudyDocument(document.id);
      const nextDocuments = documents.filter((item) => item.id !== document.id);
      setDocuments(nextDocuments);
      onDocumentsChanged(nextDocuments);
      setConfirmingDeleteId(null);
    } catch (requestError) {
      if (requestError.status === 401) onAuthenticationExpired();
      else setError(requestError.message || "Unable to delete this document.");
    } finally {
      setMutatingId(null);
    }
  }

  return (
    <section className="learning-page document-library-page" aria-labelledby="document-library-heading">
      <header className="learning-page-heading">
        <div>
          <span className="eyebrow">YOUR FILES</span>
          <h1 id="document-library-heading">My Uploads</h1>
          <p>View and manage PDFs you have uploaded.</p>
        </div>
        <button type="button" className="settings-back-button" onClick={onBack}>
          Back to workspace
        </button>
      </header>

      <section className="document-library" aria-label="Uploaded documents">
        <div className="document-library-toolbar">
          <label className="document-search">
            <Search size={16} aria-hidden="true" />
            <span className="visually-hidden">Search documents</span>
            <input
              type="search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search by document name"
            />
          </label>
          <label className="document-filter">
            <span>Status</span>
            <select
              value={statusFilter}
              onChange={(event) => setStatusFilter(event.target.value)}
              aria-label="Filter documents by status"
            >
              <option value="all">All</option>
              <option value="indexed">Indexed</option>
              <option value="processing">Processing</option>
              <option value="failed">Failed</option>
            </select>
          </label>
          <label className="document-filter">
            <span><ArrowDownWideNarrow size={14} aria-hidden="true" /> Sort</span>
            <select
              value={sortOrder}
              onChange={(event) => setSortOrder(event.target.value)}
              aria-label="Sort documents by upload date"
            >
              <option value="newest">Recently uploaded</option>
              <option value="oldest">Oldest first</option>
            </select>
          </label>
        </div>

        {error && <p className="learning-error" role="alert">{error}</p>}

        {loading ? (
          <p className="learning-state" role="status">Loading documents…</p>
        ) : documents.length === 0 ? (
          <div className="document-library-empty">
            <FileText size={28} aria-hidden="true" />
            <h2>No uploads yet</h2>
            <p>PDFs you add in Surface, Study, Quiz, or Flashcards will be listed here.</p>
          </div>
        ) : visibleDocuments.length === 0 ? (
          <div className="document-library-empty">
            <Search size={25} aria-hidden="true" />
            <h2>No matching documents</h2>
            <p>Try a different name or status filter.</p>
          </div>
        ) : (
          <ul className="document-library-list">
            {visibleDocuments.map((document) => (
              <li className="document-library-row" key={document.id}>
                <span className="document-library-icon" aria-hidden="true">
                  <FileText size={20} />
                </span>
                <div className="document-library-main">
                  {editingId === document.id ? (
                    <form
                      className="document-rename-form"
                      onSubmit={(event) => {
                        event.preventDefault();
                        saveName(document);
                      }}
                    >
                      <label>
                        <span className="visually-hidden">New PDF filename</span>
                        <input
                          autoFocus
                          maxLength={255}
                          value={nameDraft}
                          onChange={(event) => setNameDraft(event.target.value)}
                          aria-label="New PDF filename"
                          disabled={mutatingId === document.id}
                        />
                      </label>
                      <button type="submit" disabled={!nameDraft.trim() || mutatingId === document.id}>
                        {mutatingId === document.id ? "Saving…" : "Save"}
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setEditingId(null);
                          setNameDraft("");
                        }}
                        disabled={mutatingId === document.id}
                      >
                        Cancel
                      </button>
                    </form>
                  ) : (
                    <>
                      <strong className="document-library-name" title={document.filename}>
                        {document.filename}
                      </strong>
                      <div className="document-library-meta">
                        <span>PDF</span>
                        <span>{formatFileSize(document.size_bytes)}</span>
                        <span>{document.page_count} {document.page_count === 1 ? "page" : "pages"}</span>
                        <time dateTime={document.created_at}>
                          {formatUploadDate(document.created_at)}
                        </time>
                      </div>
                    </>
                  )}
                  <span className={`document-library-status ${document.status}`}>
                    {document.status === "indexed"
                      ? "Indexed"
                      : document.status === "processing"
                        ? "Processing"
                        : "Failed"}
                  </span>
                  {document.status_message && (
                    <small className="document-library-status-message">
                      {document.status_message}
                    </small>
                  )}
                </div>
                <div className="document-library-actions">
                  <button
                    type="button"
                    className="document-action-button"
                    onClick={() => openDocument(document)}
                    disabled={mutatingId !== null || fileAction !== null}
                    aria-label={`View ${document.filename}`}
                    title="View PDF"
                  >
                    {fileAction?.id === document.id && fileAction.type === "view"
                      ? <LoaderCircle size={16} className="flashcard-upload-spinner" aria-hidden="true" />
                      : <Eye size={16} aria-hidden="true" />}
                    <span>View</span>
                  </button>
                  <button
                    type="button"
                    className="document-action-button"
                    onClick={() => downloadDocument(document)}
                    disabled={mutatingId !== null || fileAction !== null}
                    aria-label={`Download ${document.filename}`}
                    title="Download PDF"
                  >
                    {fileAction?.id === document.id && fileAction.type === "download"
                      ? <LoaderCircle size={16} className="flashcard-upload-spinner" aria-hidden="true" />
                      : <Download size={15} aria-hidden="true" />}
                    <span>Download</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setEditingId(document.id);
                      setNameDraft(document.filename);
                      setConfirmingDeleteId(null);
                    }}
                    disabled={mutatingId !== null || fileAction !== null}
                    aria-label={`Rename ${document.filename}`}
                    title="Rename document"
                  >
                    <Pencil size={15} aria-hidden="true" />
                    <span>Rename</span>
                  </button>
                  {confirmingDeleteId === document.id ? (
                    <div className="document-delete-confirm" role="alertdialog" aria-label={`Confirm deleting ${document.filename}`}>
                      <span>Delete this PDF?</span>
                      <button
                        type="button"
                        onClick={() => removeDocument(document)}
                        disabled={mutatingId === document.id}
                      >
                        {mutatingId === document.id ? "Deleting…" : "Confirm"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setConfirmingDeleteId(null)}
                        disabled={mutatingId === document.id}
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <button
                      type="button"
                      className="document-delete-button"
                      onClick={() => {
                        setConfirmingDeleteId(document.id);
                        setEditingId(null);
                      }}
                      disabled={mutatingId !== null || fileAction !== null}
                      aria-label={`Delete ${document.filename}`}
                      title="Delete document"
                    >
                      <Trash2 size={15} aria-hidden="true" />
                      <span>Delete</span>
                    </button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

    </section>
  );
}

export default DocumentLibrary;
