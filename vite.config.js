import { defineConfig } from "vite"

export default defineConfig({
  server: {
    host: true,
    port: 8080,
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: "index.html",
        demo: "demo/index.js",
        streaming: "demo/streaming.js",
      },
      output: {
        entryFileNames: (chunkInfo) => {
          if (chunkInfo.name === "demo") return "demo/index.js"
          if (chunkInfo.name === "streaming") return "demo/streaming.js"
          return "assets/[name]-[hash].js"
        },
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
})
