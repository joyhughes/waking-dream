#!/usr/bin/env node
/**
 * Gives every shipped model its own page, so `/waking-dream/iguanas` is a link you can send someone.
 *
 * GitHub Pages only serves files that exist, and the usual trick of copying the app to `404.html`
 * answers those links with a 404 status, which link previews and some in-app browsers refuse. So
 * each model gets a real `dist/<name>/index.html`: the same app, carrying that model's name in the
 * tags a link preview reads. The app itself picks the model out of the path.
 *
 * Driven by the manifest, so a newly exported model gets its page with no further step.
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const DIST = new URL('../dist/', import.meta.url).pathname;
const SITE_TITLE = 'Waking dream by Joyographic';
const RESERVED = new Set(['assets', 'models']);

const escapeHtml = (text) =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const app = await readFile(join(DIST, 'index.html'), 'utf8');
if (!app.includes(`<title>${SITE_TITLE}</title>`)) {
  throw new Error(`dist/index.html no longer has the title "${SITE_TITLE}", so no page could be labelled.`);
}

const { models } = JSON.parse(await readFile(join(DIST, 'models/index.json'), 'utf8'));

for (const { name, description } of models) {
  if (RESERVED.has(name)) throw new Error(`A model cannot be called "${name}": that path is the app's own.`);

  const title = escapeHtml(`${name} · ${SITE_TITLE}`);
  const tags = [
    `<title>${title}</title>`,
    `<meta property="og:title" content="${title}" />`,
    `<meta property="og:description" content="${escapeHtml(description ?? '')}" />`,
  ].join('\n    ');

  await mkdir(join(DIST, name), { recursive: true });
  await writeFile(join(DIST, name, 'index.html'), app.replace(`<title>${SITE_TITLE}</title>`, tags));
}

// A mistyped link, or one to a model that has since gone, still opens the app, which says so.
await writeFile(join(DIST, '404.html'), app);

console.log(`Wrote a share page for ${models.length} models.`);
