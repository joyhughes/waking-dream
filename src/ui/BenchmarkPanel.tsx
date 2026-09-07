import { useState } from 'react';
import type { BenchmarkRow, ModelBenchmarkRow } from '../pipeline/engine';
import { Hint } from './Controls';

/**
 * The size/speed sweep.
 *
 * Cost is close to quadratic in the capture size — every pass covers width times height pixels —
 * so the interesting question is not "is 512 slower than 256" but where on that curve the frame
 * budget runs out on this particular machine with this particular configuration. The table reports
 * what was measured; the bar is only there so the shape of the curve is visible at a glance.
 */

const SWEEP_SIZES = [128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768];

export function BenchmarkPanel({ onRun, onCompareModels }: {
  onRun: (sizes: number[]) => Promise<BenchmarkRow[]>;
  onCompareModels?: () => Promise<ModelBenchmarkRow[]>;
}) {
  const [rows, setRows] = useState<BenchmarkRow[]>([]);
  const [models, setModels] = useState<ModelBenchmarkRow[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setRows(await onRun(SWEEP_SIZES));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  const compare = async () => {
    if (!onCompareModels) return;
    setBusy(true);
    setError(null);
    try {
      setModels(await onCompareModels());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  };

  const slowest = rows.reduce((max, row) => Math.max(max, row.msPerFrame), 0);
  const slowestModel = models.reduce((max, row) => Math.max(max, row.msPerFrame), 0);

  return (
    <div className="benchmark">
      <div className="button-row">
        <button className="button" onClick={run} disabled={busy}>
          {busy ? 'Measuring…' : 'Sweep capture sizes'}
        </button>
        <Hint text="Each frame is measured with the GPU forced to finish, so these run a little pessimistic against the live counter, which overlaps frames. The shape of the curve is what to read." />
        {onCompareModels ? (
          <button className="button" onClick={compare} disabled={busy}>
            Compare models
          </button>
        ) : null}
      </div>
      {error ? <p className="error">{error}</p> : null}

      {rows.length > 0 ? (
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>Size</th>
              <th>Tensor</th>
              <th>ms</th>
              <th>fps</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.captureSize} className={row.fps >= 60 ? 'fast' : row.fps >= 30 ? 'ok' : 'slow'}>
                <td>{row.captureSize}</td>
                <td className="dim">{row.width}×{row.height}</td>
                <td>{row.msPerFrame.toFixed(2)}</td>
                <td>{row.fps.toFixed(0)}</td>
                <td className="bar-cell">
                  <span className="bar" style={{ width: `${(row.msPerFrame / slowest) * 100}%` }} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}

      {models.length > 0 ? (
        <table className="benchmark-table">
          <thead>
            <tr><th>Model</th><th>Shape</th><th>ms</th><th>fps</th><th /></tr>
          </thead>
          <tbody>
            {models.map((row) => (
              <tr key={row.label} className={row.fps >= 60 ? 'fast' : row.fps >= 30 ? 'ok' : 'slow'}>
                <td>{row.label}</td>
                <td className="dim">{row.detail}</td>
                <td>{row.msPerFrame.toFixed(2)}</td>
                <td>{row.fps.toFixed(0)}</td>
                <td className="bar-cell">
                  <span className="bar" style={{ width: `${(row.msPerFrame / slowestModel) * 100}%` }} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}


    </div>
  );
}
