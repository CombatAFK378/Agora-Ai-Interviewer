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
      "/session": { target: "http://localhost:8080", ws: true, changeOrigin: true },
      "/panel": "http://localhost:8080",
      "/health": "http://localhost:8080",
    },
  },
  build: { outDir: "dist" },
});
