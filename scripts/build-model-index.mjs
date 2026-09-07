#!/usr/bin/env node
/**
 * Regenerates `public/models/index.json` from whatever `.dnw` files are actually in that directory.
 *
 * The app reads that manifest to populate its model list, because a page cannot enumerate a
 * directory over HTTP. `train/export.py` maintains it as part of every export, but a model trained
 * in the browser and downloaded never goes through the Python exporter — so this rebuilds the
 * manifest from the files themselves, which is also the right thing to run after deleting one.
 *
 *   pnpm models:index
 */

import { readdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const MODELS_DIR = new URL('../public/models/', import.meta.url).pathname;
const MAGIC = 'DNW1';

/** Reads just the JSON header off the front of a .dnw, without loading its weights. */
function readHeader(buffer, file) {
  if (buffer.length < 8 || buffer.toString('ascii', 0, 4) !== MAGIC) {
    throw new Error(`${file} does not start with the ${MAGIC} magic.`);
  }
  const headerLength = buffer.readUInt32LE(4);
  return JSON.parse(buffer.toString('utf8', 8, 8 + headerLength));
}

const entries = (await readdir(MODELS_DIR)).filter((name) => name.endsWith('.dnw')).sort();
const models = [];
const problems = [];

for (const file of entries) {
  try {
    const buffer = await readFile(join(MODELS_DIR, file));
    const header = readHeader(buffer, file);
    const convs = (header.ops ?? []).filter((op) => op.type === 'conv');

    // Bytes on disk track cost loosely, but what actually sets the frame time is how wide the
    // residual stack is and how many blocks deep it goes — so the manifest carries both, and the
    // model list can show the tradeoff before anything is downloaded.
    const width = convs[0]?.outChannels ?? 0;
    const blocks = Math.max(0, ((header.normSlots?.length ?? 5) - 5) / 2);
    const params = (header.gpuTexels ?? 0) * 4 + (header.cpuFloats ?? 0);

    models.push({
      file,
      name: header.name ?? file.replace(/\.dnw$/, ''),
      description: header.description ?? '',
      bytes: buffer.length,
      trainedAt: header.trainedAt ?? null,
      kind: header.teacher?.kind ?? 'unknown',
      controls: (header.conditioning?.controls ?? []).map((control) => control.label),
      width,
      blocks,
      params,
    });
  } catch (error) {
    // One unreadable file should not cost you the manifest for all the others.
    problems.push(`${file}: ${error.message}`);
  }
}

models.sort((a, b) => a.name.localeCompare(b.name));
await writeFile(join(MODELS_DIR, 'index.json'), `${JSON.stringify({ models }, null, 2)}\n`);

console.log(`Indexed ${models.length} model${models.length === 1 ? '' : 's'} in public/models/`);
for (const model of models) {
  console.log(
    `  ${model.name.padEnd(24)} ${(model.bytes / 1024).toFixed(0).padStart(6)} kB  ` +
      `width ${String(model.width).padStart(3)} × ${model.blocks} blocks  ${model.controls.join(', ')}`,
  );
}
for (const problem of problems) console.warn(`  skipped ${problem}`);
