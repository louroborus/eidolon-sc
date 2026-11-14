/**
 * Zero-dependency static file server for the fetched stand assets.
 *
 * `npx http-server` needs a download, which fails without network here. This does
 * the same job for what the browser actually asks for: files under
 * corpus/stand/source/assets, with permissive CORS (the frontend loads them
 * cross-origin from :3000).
 *
 * Usage: node deploy/local-ai/rocm/serve-stands.cjs [port] [root]
 */
const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");

const port = Number(process.argv[2] || 8765);
const root = path.resolve(process.argv[3] || path.join(__dirname, "..", "..", "..", "corpus", "stand", "source", "assets"));

const TYPES = {
  ".png": "image/png",
  ".webp": "image/webp",
  ".json": "application/json",
  ".atlas": "text/plain",
  ".skel": "application/octet-stream",
  ".jpg": "image/jpeg",
};

if (!fs.existsSync(root)) {
  console.error(`[stands] root does not exist: ${root}`);
  console.error("[stands] run `eidolon stand fetch` first");
  process.exit(1);
}

const server = http.createServer((req, res) => {
  const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
  const target = path.join(root, path.normalize(urlPath).replace(/^([/\\])+/, ""));

  // Never serve outside the root.
  if (!target.startsWith(root)) {
    res.writeHead(403).end("forbidden");
    return;
  }
  fs.stat(target, (err, stat) => {
    if (err || !stat.isFile()) {
      res.writeHead(404).end("not found");
      return;
    }
    res.writeHead(200, {
      "Content-Type": TYPES[path.extname(target).toLowerCase()] || "application/octet-stream",
      "Content-Length": stat.size,
      "Access-Control-Allow-Origin": "*",
      "Cache-Control": "public, max-age=31536000, immutable",
    });
    fs.createReadStream(target).pipe(res);
  });
});

server.listen(port, "127.0.0.1", () => {
  console.log(`[stands] serving ${root} at http://127.0.0.1:${port}`);
});
