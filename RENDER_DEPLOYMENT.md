# Deploy QuantaAI to Render

This deployment uses the repository root for the Python web service and
`frontend` for the Vite static site. The separate `friend-project` and
`java-backend` folders are not part of this QuantaAI React/FastAPI deployment.

## Render services

Create a **Web Service** for the backend:

| Setting | Value |
| --- | --- |
| Root Directory | `.` (repository root) |
| Runtime | Python |
| Build Command | `pip install -r backend/requirements.txt` |
| Start Command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |

Render supplies `PORT`; do not hardcode a port or run database migrations in the
start command. `backend.main:app` is the FastAPI entry point. Its routers
provide authentication, chat, collaboration, conversations, study documents,
learning, and planner APIs. The API also exposes `/`, `/health`, and `/agents`.

Create a **Static Site** for the frontend:

| Setting | Value |
| --- | --- |
| Root Directory | `frontend` |
| Build Command | `npm ci && npm run build` |
| Publish Directory | `dist` |

Configure this frontend build environment variable:

| Variable | Value |
| --- | --- |
| `VITE_API_BASE_URL` | `https://<your-backend-service>.onrender.com` |

The value is compiled into the static bundle at build time. Set it in Render
before the first build and rebuild after changing it. The production frontend
fails fast if it is missing; the localhost default is development-only.

## Backend environment

Configure these values in the Render Web Service's Environment settings:

| Variable | Required | Value |
| --- | --- | --- |
| `DATABASE_URL` | Yes | Neon PostgreSQL connection string; store as a secret. |
| `APP_ENV` | Yes | `production` |
| `AUTH_COOKIE_SECURE` | Recommended | `true` |
| `QUANTAAI_FRONTEND_ORIGINS` | Yes | Exact frontend origin, e.g. `https://<your-frontend-service>.onrender.com`; no trailing slash. |
| `OPENROUTER_API_KEY` or `GEMINI_API_KEY` | At least one for hosted AI | Secret key for a configured online provider. |
| `QUANTAAI_AUTO_PROVIDER` | Optional | `teacher` (default) or `quanta`. |

`PORT` is supplied by Render. Do not configure it manually. The application
accepts `DATABASE_URL` directly, normalizes `postgres://`, uses Psycopg 3, and
requires TLS for Neon hosts. Prefer Neon’s pooled connection URL if the Neon
project provides one.

Instead of `DATABASE_URL`, the backend also accepts
`NEON_DB_HOST`, `NEON_DB_NAME`, `NEON_DB_PASSWORD`, and either `NEON_DB_USER` or
`NEON_DB_USERNAME`. Do not configure both forms unless you intend
`DATABASE_URL` to take precedence.

Optional backend variables:

- `OPENROUTER_MODEL` (defaults to the model in the backend), and
  `OPENROUTER_API_KEY_SECONDARY` plus `OPENROUTER_MODEL_SECONDARY` for a
  secondary provider.
- `GEMINI_MODEL` (defaults to `gemini-3.5-flash-lite`).
- `TAVILY_API_KEY` for web search and `YOUTUBE_API_KEY` for YouTube
  recommendations.
- `QUANTAAI_QUANTALM_VERSION` (defaults to `v1`). `QUANTAAI_V2_CHECKPOINT` is
  only relevant when deliberately deploying a V2 checkpoint.

Keep all API keys and database credentials in Render's secret environment
settings. Do not put them in Vite variables, source files, or committed `.env`
files. Vite variables are public in the built frontend.

## CORS, cookies, and origins

`backend.main` reads `QUANTAAI_FRONTEND_ORIGINS` and enables credentialed CORS
for those exact origins. Configure the deployed frontend origin, plus any
additional production origins that need access. Do not use `*`: authenticated
requests use cookies and CSRF headers.

The frontend and backend's default `onrender.com` hostnames are same-site, so
the existing `SameSite=Lax` session cookies can be used across the two
services. Keep HTTPS enabled and set `AUTH_COOKIE_SECURE=true`. If using custom
domains, use frontend and API hostnames under the same registrable domain
(such as `app.example.com` and `api.example.com`); unrelated sites require a
cookie/proxy design change beyond this configuration.

## Database safety and migrations

The application does not create or migrate database tables as part of web
service startup. Migrations are explicit and use the existing
`quanta_schema_migrations` ledger. The migration code has no `DROP TABLE`,
`DROP COLUMN`, or `TRUNCATE` operations; it validates existing schema and
performs the versioned schema reconciliation/additions and flashcard
normalization.

Before applying migrations to the production Neon database:

1. Review the current schema and take a Neon backup/branch.
2. Run the read-only preflight from a trusted environment with the production
   `DATABASE_URL` set:

   ```sh
   python -m backend.migrate --preflight
   ```

3. Only if the preflight passes and the changes are approved, run once as a
   Render Shell/one-off command:

   ```sh
   python -m backend.migrate
   ```

Do not run the migration on every deploy or point it at an unreviewed database.
The deployment instructions do not drop, reset, or recreate existing records.

## Uploaded PDFs and QuantaLM

Uploaded PDF bytes and extracted page content are stored in the existing Neon
`study_documents` table. Uploads do not depend on Render's ephemeral local
filesystem, so documents survive service restarts. Keep Neon storage and
connection limits in mind as users upload files.

QuantaLM is not reached through the user's laptop. The current backend loads
the V1 model in-process on the Render instance from
`quantalm/checkpoints/quantalm.pth` and `quantalm/tokenizer/vocab.json`, using
CPU inference. Ensure those artifacts are included in the deployed source.
This project does not currently configure a remote QuantaLM service endpoint.
The V2 checkpoint directory is empty in this workspace; do not set
`QUANTAAI_QUANTALM_VERSION=v2` unless a valid V2 checkpoint and matching
tokenizer artifacts are supplied. Offline generation depends on the deployed
instance's CPU and memory resources; an online provider is recommended for
production AI availability.

## Local verification

From the repository root:

```sh
python -m unittest discover -s backend/tests
```

From `frontend`:

```sh
npm test
npm run lint
npm run build
```

To check the production build with a placeholder backend origin:

```sh
VITE_API_BASE_URL=https://api.example.com npm run build
```

The frontend test and build checks do not validate live provider credentials,
Neon connectivity, the Render network, or a production browser session.
