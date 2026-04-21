#!/usr/bin/env node
/**
 * defuddle_parse.mjs — wrapper around defuddle's Node API.
 *
 * Why not the `defuddle` CLI?
 *   CLI runs with `standardize: true, removeHiddenElements: true` (hard-coded defaults).
 *   `standardize` silently deletes <section class="... ltx_figure_panel"> blocks, which
 *   arXiv latexml uses for every subsection containing a large table. Effect: papers with
 *   big benchmark tables (MiMo-Embodied, DART-GUI, …) get truncated mid-§5 and lose
 *   §Ablation / §Conclusion / Appendix.
 *
 *   Fix: pass `standardize: false, removeHiddenElements: false` via the Node API.
 *
 * Output shape matches `defuddle parse --json --md`:
 *   {title, content (markdown), wordCount, author, published, ...}  → written to <out>.json
 *   Sibling <out>.md gets the markdown alone for grep/LLM use.
 *
 * Dependencies (declared in scripts/package.json):
 *   - defuddle, linkedom — installed on first use into scripts/node_modules/ (one-time).
 *
 * Usage:  node defuddle_parse.mjs <url-or-html-path> -o <out>.json
 */
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { execSync } from "node:child_process";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

// Hard requirement: Node ≥ 18 (needs built-in fetch and top-level await).
const MAJOR = Number(process.versions.node.split(".")[0]);
if (MAJOR < 18) {
  console.error(`defuddle_parse: requires Node ≥ 18 (got ${process.versions.node}). Install a newer node (brew install node, nvm, etc.).`);
  process.exit(2);
}

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));

// One-time bootstrap: install local deps if missing. Runs `npm install` in scripts/
// where package.json lives. After first success, this is a cheap existsSync check.
if (!existsSync(`${SCRIPT_DIR}/node_modules/defuddle`) || !existsSync(`${SCRIPT_DIR}/node_modules/linkedom`)) {
  console.error("defuddle_parse: installing deps into scripts/node_modules/ (one-time) ...");
  execSync("npm install --no-fund --no-audit --silent", { cwd: SCRIPT_DIR, stdio: "inherit" });
}

const { Defuddle } = await import("defuddle/node");
const { parseHTML } = await import("linkedom");

function parseArgs(argv) {
  const args = { output: null, source: null };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "-o" || a === "--output") args.output = argv[++i];
    else if (!args.source) args.source = a;
  }
  return args;
}

async function loadHtml(source) {
  if (existsSync(source)) return { html: readFileSync(source, "utf8"), url: `file://${source}` };
  const res = await fetch(source, { headers: { "User-Agent": "Mozilla/5.0 paper-digest/1.0" } });
  if (!res.ok) throw new Error(`fetch ${source} → HTTP ${res.status}`);
  return { html: await res.text(), url: source };
}

async function main() {
  const { source, output } = parseArgs(process.argv);
  if (!source || !output) {
    console.error("usage: defuddle_parse.mjs <url-or-html-path> -o <output.json>");
    process.exit(2);
  }
  const { html, url } = await loadHtml(source);
  const { document } = parseHTML(html);
  const r = await Defuddle(document, url, {
    markdown: true, // shape matches `defuddle --json --md`: markdown sits in r.content
    standardize: false, // CRITICAL: skip the clutter pass that drops ltx_figure_panel sections
    removeHiddenElements: false, // keep collapsible appendix content
  });
  writeFileSync(output, JSON.stringify(r, null, 2));
  const mdPath = output.replace(/\.json$/i, ".md");
  if (mdPath !== output) writeFileSync(mdPath, r.content || "");
  console.error(
    `defuddle_parse: wrote ${output} (${(r.content || "").length} bytes of md, wordCount=${r.wordCount})`
  );
}

main().catch((e) => {
  console.error("defuddle_parse: FAILED —", e.message);
  process.exit(1);
});
