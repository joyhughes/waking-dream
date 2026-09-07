#!/usr/bin/env node
/**
 * Renames a `.dnw` in place — the name and description the app shows come from the header inside
 * the file, not from its filename, so two models downloaded as "pandas" and "pandas (1)" both
 * announce themselves as "pandas" until this is run.
 *
 *   node scripts/set-model-name.mjs public/models/pandas-big.dnw pandas-big "Bigger network, coarser motifs."
 */

import { readFile, writeFile } from 'node:fs/promises';

const [file, name, description] = process.argv.slice(2);
if (!file || !name) {
  console.error('usage: set-model-name.mjs <file.dnw> <name> [description]');
  process.exit(1);
}

const buffer = await readFile(file);
if (buffer.toString('ascii', 0, 4) !== 'DNW1') {
  console.error(`${file} is not a .dnw`);
  process.exit(1);
}

const headerLength = buffer.readUInt32LE(4);
const header = JSON.parse(buffer.toString('utf8', 8, 8 + headerLength));
const payload = buffer.subarray(8 + headerLength);

header.name = name;
if (description) header.description = description;

const json = Buffer.from(JSON.stringify(header), 'utf8');
// The float payload has to stay 4-byte aligned, and a longer name moves where it starts.
const padding = (-(8 + json.length)) % 4 < 0 ? (4 - ((8 + json.length) % 4)) % 4 : (4 - ((8 + json.length) % 4)) % 4;
const paddedLength = json.length + padding;

const out = Buffer.alloc(8 + paddedLength + payload.length);
out.write('DNW1', 0, 'ascii');
out.writeUInt32LE(paddedLength, 4);
json.copy(out, 8);
out.fill(0x20, 8 + json.length, 8 + paddedLength);
payload.copy(out, 8 + paddedLength);

await writeFile(file, out);
console.log(`${file} → "${name}"${description ? ` — ${description}` : ''}`);
