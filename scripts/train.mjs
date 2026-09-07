#!/usr/bin/env node
/**
 * One command for the local PyTorch training path.
 *
 * Sets up the virtualenv if it is missing, trains on whatever is in `train/styles/`, exports the
 * model into `public/models/`, and rebuilds the manifest — so the result is live on the next reload
 * of the app with nothing else to remember.
 *
 *   pnpm train -- --images ~/Pictures/photos
 *   pnpm train -- --images ~/Pictures/photos --name waves --style-size 512 --iterations 12000
 *
 * Anything not listed here is passed through to `train/style.py`, so its flags all still work.
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const trainDir = join(root, 'train');
const venv = join(trainDir, '.venv');
const python = join(venv, 'bin', 'python');

function run(command, args, options = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { stdio: 'inherit', cwd: trainDir, ...options });
    child.on('error', reject);
    child.on('close', (code) => (code === 0 ? resolvePromise() : reject(new Error(`${command} exited with ${code}`))));
  });
}

const argv = process.argv.slice(2);
function takeFlag(name) {
  const index = argv.indexOf(name);
  if (index < 0) return null;
  const value = argv[index + 1];
  argv.splice(index, 2);
  return value;
}

const images = takeFlag('--images');
const name = takeFlag('--name') ?? 'patterns';

if (!images) {
  console.error(`Missing --images.

  pnpm train -- --images ~/Pictures/photos

Content photographs are not what the network learns — that comes entirely from the style images in
train/styles/. They are only things it must keep recognisable while repainting them, so any varied
few hundred will do.`);
  process.exit(1);
}

const styles = existsSync(join(trainDir, 'styles'))
  ? (await readdir(join(trainDir, 'styles'))).filter((file) => /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(file))
  : [];

if (styles.length === 0) {
  console.error(`No style images in train/styles/.

Put a few images there — each becomes its own slider in the app. See train/styles/README.md for
what makes a good one.`);
  process.exit(1);
}

console.log(`Styles: ${styles.join(', ')}`);

if (!existsSync(python)) {
  console.log('Creating train/.venv (one time) …');
  await run('python3', ['-m', 'venv', '.venv']);
  console.log('Installing requirements (this pulls PyTorch, so it is a few minutes and a few GB) …');
  await run(python, ['-m', 'pip', 'install', '-q', '--upgrade', 'pip']);
  await run(python, ['-m', 'pip', 'install', '-r', 'requirements.txt']);
}

const runDir = join('runs', name);
await run(python, ['style.py', '--images', images, '--out', runDir, ...argv]);
await run(python, [
  'export.py',
  '--checkpoint', join(runDir, 'checkpoint.pt'),
  '--out', join('..', 'public', 'models', `${name}.dnw`),
  '--reference', join('..', 'public', 'models', 'verify.json'),
]);
await run('node', [join(root, 'scripts', 'build-model-index.mjs')], { cwd: root });

console.log(`\nDone. Reload the app — "${name}" is in the model list and loaded by default.`);
