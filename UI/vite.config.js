import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    rollupOptions: {
      input: "public/index.html",
      output: {
        entryFileNames: "index.js", // force the JS name
        assetFileNames: "assets/[name]-[hash][extname]", // CSS/images stay in assets
      },
    },
  },
});
