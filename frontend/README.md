# React + Vite

## FastAPI connection

The frontend reads `VITE_API_BASE_URL` from Vite's environment. Copy
`.env.example` to `.env.local` for local development and update it when
FastAPI is running at a different address. Production builds require
`VITE_API_BASE_URL` to point to the deployed FastAPI origin; it is compiled
into the static bundle.

Run the FastAPI backend separately, then start the frontend with
`npm run dev -- --host localhost`. Keep the frontend and API hostnames matched
(`localhost` with `localhost`, or `127.0.0.1` with `127.0.0.1`); the
authentication cookies use `SameSite=Lax`, so mixing those hostnames prevents
the browser from sending the session and CSRF cookies.

## Account and appearance preferences

The authenticated account menu links to the profile and Settings pages. Profile
details come from the existing `/auth/me` endpoint; fields not returned by the
API, such as the account creation date, are not displayed. Sign out uses the
CSRF-protected `/auth/logout` endpoint.

Light/dark theme and the default interaction mode are stored in browser
`localStorage`. Interaction modes remain Automatic, Online, and Offline (QuantaLM).
Online mode tries OpenRouter Primary, OpenRouter
Secondary, Google Gemini, then QuantaLM Offline; Offline uses only QuantaLM.

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
