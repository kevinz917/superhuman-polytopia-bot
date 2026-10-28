import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The studio server only accepts loopback Host/Origin headers, so the dev
// proxy presents itself as the server's own origin.
const studio = "http://127.0.0.1:8765";
const proxy = {
  target: studio,
  changeOrigin: true,
  headers: { Origin: studio },
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": proxy, "/assets": proxy } },
  // Bundles live under /app; /assets is the server's sprite cache.
  build: { outDir: "dist", assetsDir: "app", chunkSizeWarningLimit: 1200 },
});
