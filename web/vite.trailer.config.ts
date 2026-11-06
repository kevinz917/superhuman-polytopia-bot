import { defineConfig } from "vite";

// Trailer capture server: trailer.html, with trailer-public/ (games.json, the hand model, and an `assets` link to the
// local game art, artifacts/mobile-assets) as its public dir. TRAILER_PORT picks the port (default 5197).
export default defineConfig({
  publicDir: process.env.TRAILER_PUBLIC ?? "trailer-public",
  server: { port: Number(process.env.TRAILER_PORT ?? 5197), strictPort: true, host: "127.0.0.1", hmr: false },
});
