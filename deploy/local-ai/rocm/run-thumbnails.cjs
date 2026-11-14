/**
 * Run `npm run stand:thumbnails` while borrowing the installed Chrome.
 *
 * `frontend/scripts/stand-thumbnails.mts` calls `chromium.launch()` with no
 * `channel`, so Playwright looks for its own Chromium download, which is not
 * installed here and cannot be fetched (the download stalls at zero bytes).
 * Chrome is installed, and Playwright can drive it via `channel: "chrome"`.
 *
 * Rather than edit the repo's script, this shims `playwright`'s chromium.launch
 * to inject `channel: "chrome"` once, then imports the original script unchanged.
 * Nothing in the repository is modified.
 *
 *   node deploy/local-ai/rocm/run-thumbnails.cjs [--force]
 */
const path = require("node:path");
const { pathToFileURL } = require("node:url");

const frontend = path.resolve(__dirname, "..", "..", "..", "frontend");
const playwright = require(path.join(frontend, "node_modules", "playwright"));

const originalLaunch = playwright.chromium.launch.bind(playwright.chromium);
playwright.chromium.launch = (options = {}) =>
  originalLaunch({ channel: "chrome", ...options });

const script = path.join(frontend, "scripts", "stand-thumbnails.mts");
process.argv = [process.argv[0], script, ...process.argv.slice(2)];

import(pathToFileURL(script).href).catch((error) => {
  console.error(error);
  process.exit(1);
});
