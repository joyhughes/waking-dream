import { useCallback, useEffect, useRef, useState } from 'react';
import {
  DEFAULT_CONFIG,
  Engine,
  type BenchmarkRow,
  type EngineConfig,
  type EngineStatus,
  type ProcessorMode,
} from './pipeline/engine';
import { CanvasRecorder, download, saveCanvasFrame, timestampedName } from './pipeline/recorder';
import { CameraSource, ImageSource, VideoFileSource, type FrameSource } from './pipeline/sources';
import { createTestPattern } from './pipeline/testPattern';
import type { ControlSpec } from './model/format';
import type { FeatureBank } from './model/shallowDream';
import { BenchmarkPanel } from './ui/BenchmarkPanel';
import { VideoTransport } from './ui/VideoTransport';
import { ButtonRow, Choice, FileButton, Section, Slider, Toggle } from './ui/Controls';

/**
 * Capture sizes offered as one-click presets. The slider covers everything between; these are the
 * sizes worth comparing directly, spaced so each step is roughly a doubling of pixel count.
 */
const SIZE_PRESETS = [128, 192, 256, 384, 512, 768];

/** How often the status readout refreshes. The engine reports every frame; re-rendering React sixty
 *  times a second to move a decimal point would cost more than the pipeline being measured. */
const STATUS_INTERVAL_MS = 200;

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<Engine | null>(null);
  const recorderRef = useRef<CanvasRecorder | null>(null);
  const latestStatus = useRef<EngineStatus | null>(null);

  const [config, setConfigState] = useState<EngineConfig>(DEFAULT_CONFIG);
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [fatal, setFatal] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [sourceLabel, setSourceLabel] = useState<string>('none');
  const [activeSource, setActiveSource] = useState<FrameSource | null>(null);
  const [modelControls, setModelControls] = useState<ControlSpec[]>([]);
  const [modelInfo, setModelInfo] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let engine: Engine;
    try {
      engine = new Engine(canvas);
    } catch (error) {
      setFatal(error instanceof Error ? error.message : String(error));
      return;
    }

    engineRef.current = engine;
    recorderRef.current = new CanvasRecorder(canvas);
    engine.onStatus = (next) => {
      latestStatus.current = next;
    };
    engine.start();

    const interval = window.setInterval(() => {
      if (latestStatus.current) setStatus(latestStatus.current);
    }, STATUS_INTERVAL_MS);

    return () => {
      window.clearInterval(interval);
      engine.dispose();
      engineRef.current = null;
    };
  }, []);

  const patchConfig = useCallback((patch: Partial<EngineConfig>) => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.setConfig(patch);
    setConfigState(engine.getConfig());
  }, []);

  const attachSource = useCallback(
    async (open: () => Promise<FrameSource>) => {
      const engine = engineRef.current;
      if (!engine) return;
      setNotice(null);
      try {
        const source = await open();
        engine.setSource(source);
        setSourceLabel(source.label);
        setActiveSource(source);
        engine.start();
      } catch (error) {
        setNotice(error instanceof Error ? error.message : String(error));
      }
    },
    [],
  );

  const openCamera = useCallback(
    async (deviceId?: string) => {
      await attachSource(() => CameraSource.open(deviceId));
      // Device labels are empty until a camera permission has been granted at least once, so the
      // list is only worth populating after the first successful open.
      setCameras(await CameraSource.listCameras());
    },
    [attachSource],
  );

  const loadModelFile = useCallback(async (file: File) => {
    const engine = engineRef.current;
    if (!engine) return;
    setNotice(null);
    try {
      const model = engine.loadModelFromBuffer(await file.arrayBuffer());
      setModelControls(model.controls);
      setModelInfo(
        `${model.name} · ${(model.byteLength / 1024).toFixed(0)} kB` +
          (model.trainedAt ? ` · trained at ${model.trainedAt}px` : ''),
      );
      setConfigState(engine.getConfig());
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    }
  }, []);

  const runBenchmark = useCallback(async (sizes: number[]): Promise<BenchmarkRow[]> => {
    const engine = engineRef.current;
    if (!engine) throw new Error('The engine is not running.');
    const rows = await engine.benchmark(sizes);
    setConfigState(engine.getConfig());
    return rows;
  }, []);

  const toggleRecording = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder) return;
    try {
      if (recorder.recording) {
        const blob = await recorder.stop();
        download(blob, timestampedName('dreamnet', blob.type.includes('mp4') ? 'mp4' : 'webm'));
        setRecording(false);
      } else {
        recorder.start();
        setRecording(true);
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
      setRecording(false);
    }
  }, []);

  if (fatal) {
    return (
      <div className="fatal">
        <h1>DreamNet cannot start</h1>
        <p>{fatal}</p>
        <p className="note">
          The runtime needs WebGL2 with renderable float textures. Every current desktop browser has
          both; a very old GPU or a virtualized one may not.
        </p>
      </div>
    );
  }

  const timing = status?.timing;

  return (
    <div className="app">
      <main className="stage">
        <canvas ref={canvasRef} />
        {sourceLabel === 'none' ? (
          <div className="empty-stage">
            <h1>DreamNet</h1>
            <p>Real-time dream filtering. Pick a source to begin.</p>
            <ButtonRow>
              <button className="button primary" onClick={() => void openCamera()}>
                Use camera
              </button>
              <FileButton label="Open video" accept="video/*" onFile={(file) => void attachSource(() => VideoFileSource.open(file))} />
              <FileButton label="Open image" accept="image/*" onFile={(file) => void attachSource(() => ImageSource.open(file))} />
              <button
                className="button"
                data-testid="test-pattern"
                onClick={() => void attachSource(() => ImageSource.fromCanvas(createTestPattern(), 'test pattern'))}
              >
                Test pattern
              </button>
            </ButtonRow>
          </div>
        ) : null}
        {status?.error ? <div className="stage-error">{status.error}</div> : null}
      </main>

      <aside className="panel">
        <header className="panel-header">
          <h1>DreamNet</h1>
          <div className="readout">
            <span className={timing && timing.fps >= 50 ? 'fast' : timing && timing.fps >= 25 ? 'ok' : 'slow'}>
              {timing ? timing.fps.toFixed(0) : '–'} fps
            </span>
            <span className="dim">
              {status ? `${status.captureWidth}×${status.captureHeight}` : '–'}
            </span>
            <span className="dim">
              {timing?.gpuMs != null ? `${timing.gpuMs.toFixed(2)} ms gpu` : `${timing?.cpuMs.toFixed(2) ?? '–'} ms cpu`}
            </span>
          </div>
        </header>

        {notice ? <p className="notice">{notice}</p> : null}

        <Section title="Source" hint={sourceLabel}>
          <ButtonRow>
            <button className="button" onClick={() => void openCamera()}>Camera</button>
            <FileButton label="Video…" accept="video/*" onFile={(file) => void attachSource(() => VideoFileSource.open(file))} />
            <FileButton label="Image…" accept="image/*" onFile={(file) => void attachSource(() => ImageSource.open(file))} />
            <button
              className="button"
              onClick={() => void attachSource(() => ImageSource.fromCanvas(createTestPattern(), 'test pattern'))}
              title="A deterministic synthetic frame. The honest way to compare two settings."
            >
              Test pattern
            </button>
          </ButtonRow>
          {cameras.length > 1 ? (
            <Choice
              label="Camera device"
              value={''}
              options={[{ value: '', label: 'Switch to…' }, ...cameras.map((device) => ({
                value: device.deviceId,
                label: device.label || 'Camera',
              }))]}
              onChange={(deviceId) => {
                if (deviceId) void openCamera(deviceId);
              }}
            />
          ) : null}
          <Toggle
            label="Mirror"
            checked={config.mirror ?? engineRef.current?.currentSource?.defaultMirror ?? false}
            onChange={(mirror) => patchConfig({ mirror })}
          />
          {activeSource instanceof VideoFileSource ? <VideoTransport source={activeSource} /> : null}
        </Section>

        <Section title="Processor" hint={config.processor}>
          <Choice<ProcessorMode>
            label="Mode"
            value={config.processor}
            options={[
              { value: 'off', label: 'Passthrough' },
              { value: 'shallow', label: 'Shallow ascent (no model)' },
              { value: 'model', label: 'DreamNet model', disabled: modelControls.length === 0 && !modelInfo },
            ]}
            onChange={(processor) => patchConfig({ processor })}
          />
          <ButtonRow>
            <FileButton label="Load .dnw model…" accept=".dnw" onFile={(file) => void loadModelFile(file)} />
          </ButtonRow>
          {modelInfo ? <p className="note">{modelInfo}</p> : (
            <p className="note">
              No trained model loaded. Train one with <code>train/</code>, or stay on shallow ascent —
              it runs the same octaves and feedback with a hand-built one-layer feature bank.
            </p>
          )}
        </Section>

        <Section title="Capture size" hint={`${config.captureSize}px`}>
          <ButtonRow>
            {SIZE_PRESETS.map((size) => (
              <button
                key={size}
                className={`button small ${config.captureSize === size ? 'active' : ''}`}
                onClick={() => patchConfig({ captureSize: size })}
              >
                {size}
              </button>
            ))}
          </ButtonRow>
          <Slider
            label="Longest side"
            value={config.captureSize}
            min={64}
            max={1024}
            step={8}
            onChange={(captureSize) => patchConfig({ captureSize })}
            format={(value) => `${value} px`}
            title="The resolution the network actually sees. Cost grows with its square."
          />
          <BenchmarkPanel onRun={runBenchmark} />
        </Section>

        {config.processor === 'shallow' ? (
          <Section title="Shallow ascent">
            <Choice<FeatureBank>
              label="Feature bank"
              value={config.shallow.bank}
              options={[
                { value: 'gabor', label: 'Gabor — flow and ridges' },
                { value: 'blob', label: 'Blob — cells and dots' },
                { value: 'random', label: 'Random — mixed texture' },
              ]}
              onChange={(bank) => patchConfig({ shallow: { ...config.shallow, bank } })}
            />
            <Slider
              label="Step size"
              value={config.shallow.stepSize}
              min={0}
              max={0.4}
              step={0.005}
              onChange={(stepSize) => patchConfig({ shallow: { ...config.shallow, stepSize } })}
            />
            <Slider
              label="Steps per octave"
              value={config.shallow.steps}
              min={1}
              max={8}
              step={1}
              onChange={(steps) => patchConfig({ shallow: { ...config.shallow, steps } })}
            />
            <Slider
              label="Octaves"
              value={config.shallow.octaves}
              min={1}
              max={5}
              step={1}
              onChange={(octaves) => patchConfig({ shallow: { ...config.shallow, octaves } })}
            />
            <Slider
              label="Octave scale"
              value={config.shallow.octaveScale}
              min={1.2}
              max={2.5}
              step={0.05}
              onChange={(octaveScale) => patchConfig({ shallow: { ...config.shallow, octaveScale } })}
            />
            <Slider
              label="Filters"
              value={config.shallow.filters}
              min={4}
              max={64}
              step={4}
              onChange={(filters) => patchConfig({ shallow: { ...config.shallow, filters } })}
            />
            <Slider
              label="Filter size"
              value={config.shallow.kernel}
              min={3}
              max={9}
              step={2}
              onChange={(kernel) => patchConfig({ shallow: { ...config.shallow, kernel } })}
              format={(value) => `${value}×${value}`}
            />
            <Slider
              label="Bank seed"
              value={config.shallow.seed}
              min={0}
              max={64}
              step={1}
              onChange={(seed) => patchConfig({ shallow: { ...config.shallow, seed } })}
            />
            <ButtonRow>
              <button
                className="button"
                onClick={() => patchConfig({ shallow: { ...config.shallow, seed: Math.floor(Math.random() * 64) } })}
                title="Rerolls the filter bank. Same statistics, different filters, different look."
              >
                Reroll bank
              </button>
              <button
                className="button"
                onClick={() => patchConfig({ shallow: DEFAULT_CONFIG.shallow })}
              >
                Reset ascent
              </button>
            </ButtonRow>
          </Section>
        ) : null}

        {config.processor === 'model' && modelControls.length > 0 ? (
          <Section title="Model controls" hint="conditioned">
            {modelControls.map((control, index) => (
              <Slider
                key={control.name}
                label={control.label}
                title={control.description}
                value={config.modelControls[index] ?? control.default}
                min={control.min}
                max={control.max}
                step={(control.max - control.min) / 100}
                onChange={(value) => {
                  const next = config.modelControls.slice();
                  next[index] = value;
                  patchConfig({ modelControls: next });
                }}
              />
            ))}
          </Section>
        ) : null}

        <Section title="Feedback" hint={config.feedback.enabled ? 'on' : 'off'}>
          <Toggle
            label="Feed the output back in"
            checked={config.feedback.enabled}
            onChange={(enabled) => patchConfig({ feedback: { ...config.feedback, enabled } })}
            title="Turns a per-frame filter into a recursion that keeps evolving."
          />
          <Slider
            label="Live frame"
            value={config.feedback.source}
            min={0}
            max={1}
            step={0.01}
            onChange={(source) => patchConfig({ feedback: { ...config.feedback, source } })}
          />
          <Slider
            label="Previous output"
            value={config.feedback.previous}
            min={0}
            max={1}
            step={0.01}
            onChange={(previous) => patchConfig({ feedback: { ...config.feedback, previous } })}
          />
          <Slider
            label="Zoom per frame"
            value={config.feedback.zoom}
            min={0.97}
            max={1.05}
            step={0.001}
            onChange={(zoom) => patchConfig({ feedback: { ...config.feedback, zoom } })}
            format={(value) => `${((value - 1) * 100).toFixed(1)}%`}
          />
          <Slider
            label="Rotate per frame"
            value={config.feedback.rotate}
            min={-1}
            max={1}
            step={0.01}
            onChange={(rotate) => patchConfig({ feedback: { ...config.feedback, rotate } })}
            format={(value) => `${value.toFixed(2)}°`}
          />
          <Slider
            label="Drift x"
            value={config.feedback.driftX}
            min={-0.02}
            max={0.02}
            step={0.001}
            onChange={(driftX) => patchConfig({ feedback: { ...config.feedback, driftX } })}
          />
          <Slider
            label="Drift y"
            value={config.feedback.driftY}
            min={-0.02}
            max={0.02}
            step={0.001}
            onChange={(driftY) => patchConfig({ feedback: { ...config.feedback, driftY } })}
          />
          <Slider
            label="Fade"
            value={config.feedback.fade}
            min={0.9}
            max={1}
            step={0.002}
            onChange={(fade) => patchConfig({ feedback: { ...config.feedback, fade } })}
            title="Damps what comes back around. Below 1 the recursion cannot run away."
          />
          <ButtonRow>
            <button className="button" onClick={() => engineRef.current?.discardFeedback()}>
              Reset recursion
            </button>
          </ButtonRow>
        </Section>

        <Section title="Display" defaultOpen={false}>
          <Slider
            label="Effect amount"
            value={config.display.mix}
            min={0}
            max={1}
            step={0.01}
            onChange={(mix) => patchConfig({ display: { ...config.display, mix } })}
          />
          <Slider
            label="Gain"
            value={config.display.gain}
            min={0.4}
            max={2}
            step={0.01}
            onChange={(gain) => patchConfig({ display: { ...config.display, gain } })}
          />
          <Slider
            label="Saturation"
            value={config.display.saturation}
            min={0}
            max={2}
            step={0.01}
            onChange={(saturation) => patchConfig({ display: { ...config.display, saturation } })}
          />
        </Section>

        <Section title="Capture" defaultOpen={false}>
          <ButtonRow>
            <button
              className={`button ${recording ? 'recording' : ''}`}
              onClick={() => void toggleRecording()}
              disabled={!CanvasRecorder.supported}
            >
              {recording ? 'Stop recording' : 'Record video'}
            </button>
            <button
              className="button"
              onClick={() => {
                const canvas = canvasRef.current;
                if (canvas) void saveCanvasFrame(canvas, timestampedName('dreamnet', 'png'));
              }}
            >
              Save frame
            </button>
          </ButtonRow>
          {!CanvasRecorder.supported ? <p className="note">This browser cannot record canvas video.</p> : null}
        </Section>

        <Section title="Settings" defaultOpen={false}>
          <ButtonRow>
            <button
              className="button"
              onClick={() => {
                patchConfig({ ...DEFAULT_CONFIG, modelControls: config.modelControls });
                engineRef.current?.discardFeedback();
              }}
            >
              Reset everything
            </button>
            <button
              className="button"
              onClick={() => {
                void navigator.clipboard?.writeText(JSON.stringify(config, null, 2));
                setNotice('Settings copied to the clipboard.');
              }}
              title="Copies the whole parameter set as JSON, so a look you found can be written down."
            >
              Copy settings
            </button>
            <button
              className="button"
              onClick={() => {
                void navigator.clipboard
                  ?.readText()
                  .then((text) => {
                    patchConfig(JSON.parse(text) as Partial<EngineConfig>);
                    setNotice(null);
                  })
                  .catch(() => setNotice('Could not read a settings JSON from the clipboard.'));
              }}
            >
              Paste settings
            </button>
          </ButtonRow>
        </Section>

        <footer className="panel-footer">
          <div>{status?.renderer ?? ''}</div>
          <div className="dim">
            {status ? `${status.programCount} shaders · ${status.poolMegabytes.toFixed(1)} MB textures` : ''}
            {status && !status.supportsGpuTiming ? ' · no GPU timer' : ''}
          </div>
        </footer>
      </aside>
    </div>
  );
}
