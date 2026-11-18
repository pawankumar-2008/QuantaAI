# FastAPI authentication

The React application and its seven-agent chat API use `backend.main:app` as their
single authentication owner. The separate `friend-project` application is not
used by this frontend and retains its own legacy authentication. That standalone
app is not integrated with these accounts or sessions; its existing secret and
browser-token handling must be hardened before exposing it publicly.

## PostgreSQL

The active FastAPI application uses one cached SQLAlchemy engine for
authentication, conversation history, and migrations. Set either `DATABASE_URL`
or all four `NEON_DB_*` variables. `DATABASE_URL` takes precedence when both
forms are present:

- `NEON_DB_HOST`
- `NEON_DB_NAME`
- `NEON_DB_USER` (legacy `NEON_DB_USERNAME` is also accepted)
- `NEON_DB_PASSWORD`

For Neon, configure either:

- `DATABASE_URL` with a PostgreSQL URL, for example
	`postgresql+psycopg://USER:PASSWORD@HOST/DATABASE?sslmode=require`.
- `NEON_DB_HOST`, `NEON_DB_NAME`, `NEON_DB_USERNAME`, and `NEON_DB_PASSWORD`.

The legacy `postgres://` scheme is normalized to PostgreSQL with the installed
Psycopg 3 driver. Neon hosts always use TLS (`sslmode=require`); an explicit
`verify-ca` or `verify-full` mode is preserved. PostgreSQL connections have a
5-second connection timeout, a 5-second pool checkout timeout, and pool
pre-ping. SQLite URLs remain supported for local/test configuration; automated
tests use isolated in-memory SQLite databases through FastAPI dependency
overrides.

The active backend's database driver is declared in
`backend/requirements-auth.txt` as `psycopg[binary]>=3.2`. The complete
FastAPI runtime dependencies, including the ASGI server and in-process QuantaLM
runtime, are listed in `backend/requirements.txt`. Install them in the existing
virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
```

To configure Neon for the current PowerShell session, set values locally and do
not commit or paste credentials into source control:

```powershell
$env:NEON_DB_HOST = "<your-neon-host>"
$env:NEON_DB_NAME = "<your-database-name>"
$env:NEON_DB_USER = "<your-username>"
$env:NEON_DB_PASSWORD = "<your-password>"
```

Or set `DATABASE_URL` instead, with URL-encoded credentials if they contain
reserved URL characters. A credential-free variable template is available in
the workspace root as `.env.example`. FastAPI loads the workspace-root `.env`
when its backend configuration is imported. Existing process environment
variables take precedence and are never overwritten by values from that file.
Keep the standalone `friend-project` settings in `friend-project/.env`, outside
its public `static/` directory.

After reviewing the schema and backing up the target database, apply the
reconciliation migration from the workspace root:

```powershell
.\.venv\Scripts\python.exe -m backend.migrate
```

`backend/migrate.py` applies the explicit reconciliation migration
`backend/migrations/v003_fastapi_schema_reconciliation.py` and the additive
`backend/migrations/v004_study_documents.py` and
`backend/migrations/v005_learning_and_planner.py` and
`backend/migrations/v006_document_original_files.py` and
`backend/migrations/v007_normalized_flashcards.py` and
`backend/migrations/v008_planner_task_descriptions.py`, inside one database
transaction. The latest migration adds a nullable description column to the existing
`planner_tasks` table without replacing task data. It validates existing objects before recording each version and never runs
legacy SQL migrations 001 or 002 automatically. It reuses the existing `users`,
`conversations`, and `messages` tables, adds `profile` to `users` only when
missing, and creates `auth_sessions`, `study_documents`, and the migration
ledger only when missing. Incompatible, partial, orphaned, or duplicate-email
schemas are rejected before the migration is recorded. Do not execute the
migration until the target Neon schema has been reviewed and backed up.

Set `AUTH_COOKIE_SECURE=true` when serving the API over HTTPS. The default local
development configuration uses host-only, `SameSite=Lax` cookies. The session
cookie is HttpOnly; state-changing browser requests also require the CSRF token
returned by `GET /auth/csrf`.

In production, serve the React frontend and API from the same site (or through a
same-site reverse proxy) so `SameSite=Lax` cookies are sent. Set
`QUANTAAI_FRONTEND_ORIGINS` to the exact comma-separated frontend origins and
configure `VITE_API_BASE_URL` for the deployed API.

## AI provider and Study Agent resources

Online mode and `auto` mode try providers in order: OpenRouter Primary
(`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`), OpenRouter Secondary
(`OPENROUTER_API_KEY_SECONDARY`, `OPENROUTER_MODEL_SECONDARY`), Google Gemini
(`GEMINI_API_KEY`, `GEMINI_MODEL`), then local QuantaLM. The secondary OpenRouter
provider is unavailable unless its own key and model are configured; it never
reuses the primary key. Gemini defaults to `gemini-3.5-flash-lite`; set
`GEMINI_MODEL` to override it. Provider failures advance once to the next
configured provider; QuantaLM output is returned with a reliability warning
when its quality check flags it. Explicit Offline mode uses only local QuantaLM
and never tries an online provider. Study requests are supported offline, with
or without a selected PDF.

Study Agent YouTube recommendations are optional. Set `YOUTUBE_API_KEY` and
enable YouTube Data API v3 for that key. The key is used only by the backend.
Missing or invalid keys, quota exhaustion, network errors, and empty results are
reported with the Study response. `GET /health` reports whether a YouTube key is
configured; it does not validate the key or quota. FastAPI loads the workspace-
root `.env` without overriding values already set in the process environment.

## Endpoints

- `GET /auth/csrf` - issue/read the CSRF token.
- `POST /auth/register` - create an account and authenticated session.
- `POST /auth/login` - verify credentials and create an authenticated session.
- `GET /auth/me` - return the current authenticated user.
- `PATCH /auth/me` - update the current user's name and profile preferences.
- `POST /auth/logout` - revoke the current session.
- `POST /chat` - requires a valid session cookie and CSRF header.
- `POST /collaboration/synthesize` - synthesizes authenticated, successful agent
  contributions; offline mode uses only local QuantaLM.
- `POST /conversations` - create an authenticated user's conversation.
- `GET /conversations` - list the current user's conversations.
- `GET /conversations/{id}` - read a conversation and paginated messages.
- `PATCH /conversations/{id}` - rename an owned conversation.
- `DELETE /conversations/{id}` - delete an owned conversation and its messages.
- `POST /conversations/{id}/messages` - append attributed messages atomically.
- `GET /study-documents` - list the authenticated user's uploaded PDFs.
- `POST /study-documents` - upload and index a PDF using the multipart `file`
  field; requires the CSRF header.
- `DELETE /study-documents/{document_id}` - remove an owned PDF; requires the
  CSRF header.
- `GET /planner/tasks` - list the authenticated user's planner tasks.
- `POST /planner/tasks` - create a task with a title, optional description, and
  due date; requires the CSRF header. Dated schedules generated by the Planner
  Agent are generated as validated structured data using the backend's current
  Asia/Kolkata date and saved through this same endpoint. The first schedule
  portion is bounded to 7 days by default and at most 14 days; the frontend
  skips matching title/date tasks and reports successful and failed saves.
- `PATCH /planner/tasks/{task_id}` - update an owned task; requires the CSRF header.
- `DELETE /planner/tasks/{task_id}` - remove an owned task; requires the CSRF header.
- `POST /learning/quizzes` and `POST /learning/quizzes/{quiz_id}/submit` -
  generate a topic- or document-based quiz and persist its result.
- `GET /learning/quizzes` - list the authenticated user's quiz history.
- `POST /learning/flashcards` - generate and save topic- or document-based
  flashcards. The optional `difficulty` is `basic` or `medium` (default:
  `basic`); generation analyzes the topic and grounds document-based cards in
  the selected PDF. Deck responses include the saved difficulty when recorded.
- `GET`, `PATCH`, and `DELETE /learning/flashcards/{deck_id}` - retrieve or
  manage an owned flashcard deck and its study progress.
- `GET /learning/progress` - summarize quiz performance, weak topics, and
  recent learning activity.

## PDF uploads and document-grounded answers

The Study Agent and Surface AI workspaces support PDFs up to 10 MB and 200
pages. The backend extracts selectable page text (scanned image-only PDFs and
password-protected PDFs are not supported), stores the document pages in the
existing `study_documents` table, and reports `processing`, `indexed`, or
`failed` status. Upload responses include the processing result and a safe
status message; documents are scoped to their authenticated owner.

When an indexed document is selected, Study requests retrieve relevant
page-aware text chunks with the existing TF-IDF RAG helper and include bounded
excerpts in the Study Agent prompt. Surface AI stores its selected PDF
references in the current conversation's existing message metadata and retrieves
the latest attachment set on follow-up questions. Attachments are validated
against the authenticated user and do not carry over to another conversation.
The excerpts are treated as untrusted reference data, with page numbers
available for citation. Normal Study and Surface conversations without PDFs
remain unchanged. The feature does not add a separate embedding model, vector
database, or database schema.

The browser sends session credentials only as cookies; password hashes and
access tokens are never returned in API response bodies.

FastAPI maps its conversation model onto the existing `conversations` and
`messages` tables. The historical `001_auth_tables.sql` and
`002_conversation_history.sql` files use a different schema and must not be
executed manually for this database. Every conversation remains scoped to its
owning `users.user_id`.
