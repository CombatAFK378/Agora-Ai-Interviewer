import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, run `npm run dev` (Vite on :5173) alongside the media worker (:8080).
// The proxy forwards API calls and the events WebSocket to the worker, so the
// browser talks to a single origin (:5173) just like it will in production —
// where the worker itself serves the built dist/ and the API from one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // /session carries the events WebSocket as well as the REST calls.
      "/session": { target: "http://localhost:8080", ws: true, changeOrigin: true },
      "/panel": "http://localhost:8080",
      "/health": "http://localhost:8080",
      // These were missing, so in dev the recruiter dashboard, PDF upload and
      // the coding round all 404'd against Vite instead of reaching the worker.
      "/interviews": "http://localhost:8080",
      "/parse-pdf": "http://localhost:8080",
      "/coding": "http://localhost:8080",
      "/report": "http://localhost:8080",
      "/client-log": "http://localhost:8080",
    },
  },
  build: { outDir: "dist" },
});
