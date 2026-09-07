/**
 * A synthetic frame, so the app has something to run on before any camera or file is opened.
 *
 * It is also the honest way to compare two settings. Camera input drifts — exposure, white balance,
 * whatever moved in the room — so A/B-ing a step size against a live feed compares two different
 * inputs. This one is deterministic to the pixel, which makes it the thing to reach for when a
 * change is supposed to be small.
 *
 * The content is chosen for what the filters have to bite on: oriented edges at several angles,
 * round shapes at several sizes, a broad colour sweep, and a band of fine noise that shows
 * immediately whether a configuration is amplifying grain rather than structure.
 */
export function createTestPattern(width = 1024, height = 768): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Could not create a 2D context for the test pattern.');

  const sky = ctx.createLinearGradient(0, 0, width * 0.4, height);
  sky.addColorStop(0, '#1b2a4a');
  sky.addColorStop(0.45, '#4a4276');
  sky.addColorStop(0.75, '#b4623f');
  sky.addColorStop(1, '#efb96b');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, width, height);

  // Oriented bars: eight directions, which is what a Gabor bank is built to answer to.
  ctx.save();
  ctx.globalAlpha = 0.5;
  for (let i = 0; i < 8; i++) {
    const angle = (i * Math.PI) / 8;
    ctx.save();
    ctx.translate(width * (0.14 + 0.1 * (i % 4)), height * (i < 4 ? 0.26 : 0.72));
    ctx.rotate(angle);
    ctx.fillStyle = i % 2 ? '#f4f1e8' : '#12141c';
    for (let bar = -3; bar <= 3; bar++) {
      ctx.fillRect(-height * 0.09, bar * 11, height * 0.18, 5);
    }
    ctx.restore();
  }
  ctx.restore();

  // Discs across two octaves of size, for the blob bank and for anything that grows eyes.
  const discs: [number, number, number, string][] = [
    [0.66, 0.3, 0.15, '#f2d98c'],
    [0.82, 0.52, 0.08, '#7fd0c4'],
    [0.72, 0.68, 0.05, '#e4796a'],
    [0.9, 0.24, 0.035, '#ffffff'],
    [0.6, 0.82, 0.06, '#3f5f9c'],
  ];
  for (const [cx, cy, r, color] of discs) {
    const radius = r * Math.min(width, height);
    const shade = ctx.createRadialGradient(
      cx * width - radius * 0.3, cy * height - radius * 0.3, radius * 0.05,
      cx * width, cy * height, radius,
    );
    shade.addColorStop(0, '#ffffff');
    shade.addColorStop(0.35, color);
    shade.addColorStop(1, '#0d0f16');
    ctx.fillStyle = shade;
    ctx.beginPath();
    ctx.arc(cx * width, cy * height, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  // A strip of fine noise. Anything that turns this into structure is amplifying grain.
  const noise = ctx.createImageData(width, Math.round(height * 0.12));
  for (let i = 0; i < noise.data.length; i += 4) {
    const value = 40 + Math.random() * 180;
    noise.data[i] = value;
    noise.data[i + 1] = value;
    noise.data[i + 2] = value;
    noise.data[i + 3] = 90;
  }
  ctx.putImageData(noise, 0, Math.round(height * 0.88));

  return canvas;
}
