# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some Oxlint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and Oxlint's TypeScript related rules in your project.

## Backend integration

The dashboard no longer depends on frontend mock detection data. It loads live detections and surveys from the Member 3 FastAPI backend.

For local development:
1. Start the FastAPI backend on `http://127.0.0.1:8000`.
2. Start this Vite frontend with `npm run dev`.
3. The Vite proxy forwards `/backend-api/*` to the FastAPI server.

The dashboard uses `GET /detections` and `GET /surveys`. Member 5 risk scoring is applied to backend detections when the backend has not stored a risk score yet.

For a deployed setup, set `VITE_API_BASE_URL` to the public FastAPI base URL.
