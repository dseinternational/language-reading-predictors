#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-or-later

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const mode = process.argv[2];

if (!["--check", "--write"].includes(mode)) {
  console.error("Usage: node scripts/format-markdown.mjs --check|--write");
  process.exit(2);
}

const trackedMarkdown = spawnSync("git", ["ls-files", "--", "*.md", ":(exclude)data/**/*.md"], {
  encoding: "utf8",
});

if (trackedMarkdown.status !== 0) {
  if (trackedMarkdown.stderr) {
    process.stderr.write(trackedMarkdown.stderr);
  } else if (trackedMarkdown.error) {
    console.error(trackedMarkdown.error.message);
  }
  process.exit(trackedMarkdown.status ?? 1);
}

const files = trackedMarkdown.stdout.split(/\r?\n/u).filter((file) => file && existsSync(file));

if (files.length === 0) {
  process.exit(0);
}

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "..");
const prettierBin = process.platform === "win32" ? "prettier.cmd" : "prettier";
const localPrettier = path.join(repoRoot, "node_modules", ".bin", prettierBin);
const prettier = existsSync(localPrettier) ? localPrettier : prettierBin;
// Keep batches below cmd.exe's 8,191-character limit, with room for the command.
// Check every batch and retain any failure in the final exit status.
const MAX_ARGS_LENGTH = 6000;
const batches = [];
let batch = [];
let batchLength = 0;

for (const file of files) {
  if (batch.length > 0 && batchLength + file.length + 1 > MAX_ARGS_LENGTH) {
    batches.push(batch);
    batch = [];
    batchLength = 0;
  }
  batch.push(file);
  batchLength += file.length + 1;
}
if (batch.length > 0) {
  batches.push(batch);
}

let exitStatus = 0;

for (const chunk of batches) {
  const result = spawnSync(prettier, [mode, ...chunk], {
    shell: process.platform === "win32",
    stdio: "inherit",
  });
  if ((result.status ?? 1) !== 0) {
    exitStatus = result.status ?? 1;
  }
}

process.exit(exitStatus);
