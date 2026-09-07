import { useCallback, useEffect, useRef, useState } from 'react';
import {
  DEFAULT_CONFIG,
  Engine,
  type BenchmarkRow,
  type EngineConfig,
  type EngineStatus,
  type ModelBenchmarkRow,
  type ProcessorMode,
} from './pipeline/engine';
import { CanvasRecorder, download, extensionForMimeType, saveCanvasFrame, timestampedName } from './pipeline/recorder';
import { buildParameters, couldCarryParameters, describeParameters, readParameters, type EmbeddedParameters } from './pipeline/parameters';
import { AudioAnalyser, FREQUENCY_BANDS } from './pipeline/audio';
import { CameraSource, ImageSource, VideoFileSource, type CameraFacing, type FrameSource } from './pipeline/sources';
import { createTestPattern } from './pipeline/testPattern';
import type { ControlSpec } from './model/format';
import { costLabel, fetchModelListings, formatSize, modelUrl, type ModelListing } from './model/registry';
import { deleteSavedModel, listSavedModels, loadSavedModel, type SavedModelMeta } from './model/storage';
import { TrainPanel } from './ui/TrainPanel';
import { getDeviceLimits } from './pipeline/deviceLimits';
import type { FeatureBank } from './model/shallowDream';
import { BenchmarkPanel } from './ui/BenchmarkPanel';
import { VideoTransport } from './ui/VideoTransport';
import { ButtonRow, Choice, FileButton, ModulatedSlider, Section, Slider, Toggle } from './ui/Controls';
import type { Modulation } from './pipeline/modulation';

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
  const [listings, setListings] = useState<ModelListing[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [loadingModel, setLoadingModel] = useState(false);
  const [saved, setSaved] = useState<SavedModelMeta[]>([]);
  const [recording, setRecording] = useState(false);
  const [foundParameters, setFoundParameters] = useState<EmbeddedParameters | null>(null);
  const [immersive, setImmersive] = useState(false);
  const [cameraFacing, setCameraFacing] = useState<CameraFacing>('user');
  const [multipleCameras, setMultipleCameras] = useState(false);
  const [audioLabel, setAudioLabel] = useState<string | null>(null);

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

    // Esc, or the system gesture, exits fullscreen without going through the button.
    const onFullscreenChange = () => {
      if (!document.fullscreenElement) setImmersive(false);
    };
    document.addEventListener('fullscreenchange', onFullscreenChange);

    void listSavedModels().then(setSaved);

    void fetchModelListings().then((found) => {
      setListings(found);
      // A deployed build should open with a trained model already running, not the fallback and a
      // menu. On a phone that means the cheapest one: a 300 kB download rather than 4 MB, possibly
      // over cellular, and a network whose residual stack is a fraction of the arithmetic. Neither
      // is worth trading for a better-looking model nobody has asked for yet.
      if (found.length === 0) return;
      const pick = getDeviceLimits().preferCheapestModel
        ? [...found].sort((a, b) => (a.params ?? a.bytes) - (b.params ?? b.bytes))[0]
        : found[0];
      void loadListing(pick);
    });

    return () => {
      window.clearInterval(interval);
      document.removeEventListener('fullscreenchange', onFullscreenChange);
      engine.dispose();
      engineRef.current = null;
    };
    // loadListing is stable for the life of the engine; re-running this effect would reopen the
    // WebGL context.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const patchConfig = useCallback((patch: Partial<EngineConfig>) => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.setConfig(patch);
    setConfigState(engine.getConfig());
  }, []);

  const readParametersFromFile = useCallback(async (file: File): Promise<EmbeddedParameters | null> => {
    if (!couldCarryParameters(file)) return null;
    try {
      return readParameters(new Uint8Array(await file.arrayBuffer()));
    } catch {
      return null;
    }
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

  const openImage = useCallback(
    async (file: File) => {
      await attachSource(() => ImageSource.open(file));
      // Offered rather than applied: opening a picture to look at it is not the same as asking to
      // be moved to wherever its author had the sliders.
      setFoundParameters(await readParametersFromFile(file));
    },
    [attachSource, readParametersFromFile],
  );

  const openCamera = useCallback(
    async (deviceId?: string, facing: CameraFacing = 'user') => {
      await attachSource(() => CameraSource.open(deviceId, facing));
      // Device labels are empty until a camera permission has been granted at least once, so the
      // list is only worth populating after the first successful open.
      setCameras(await CameraSource.listCameras());
      setMultipleCameras(await CameraSource.hasMultipleCameras());
    },
    [attachSource],
  );

  /**
   * Swaps between the front and rear cameras.
   *
   * By facing direction rather than by device id: ids are unstable across sessions on iOS and their
   * labels are empty until permission has been granted, so the device list is useless on the very
   * first open — which is exactly when someone wants to turn the camera around.
   */
  const flipCamera = useCallback(async () => {
    const next: CameraFacing = cameraFacing === 'user' ? 'environment' : 'user';
    setCameraFacing(next);
    // Back to following the source's own default, so the rear camera stops being mirrored.
    patchConfig({ mirror: null });
    await openCamera(undefined, next);
  }, [cameraFacing, openCamera, patchConfig]);

  /**
   * Hides the controls and gives the whole viewport to the picture.
   *
   * Native fullscreen is requested where it exists, but iOS Safari only grants it to video
   * elements, so the layout change has to stand on its own — and does. The two are independent:
   * the CSS gives the full viewport either way, and fullscreen additionally takes the browser
   * chrome when the browser allows it.
   */
  const toggleImmersive = useCallback(() => {
    const next = !immersive;
    setImmersive(next);
    // Immersive means the picture fills the screen, not that it is centred in a black field with
    // bars down both sides — a phone screen is far taller than any camera's aspect ratio.
    patchConfig({ fillScreen: next });
    try {
      if (next && document.fullscreenEnabled && !document.fullscreenElement) {
        void document.documentElement.requestFullscreen().catch(() => {});
      } else if (!next && document.fullscreenElement) {
        void document.exitFullscreen().catch(() => {});
      }
    } catch {
      // Fullscreen is a bonus; the layout change is the feature.
    }
  }, [immersive, patchConfig]);

  const applyLoadedModel = useCallback((model: ReturnType<Engine['loadModelFromBuffer']>, source: string) => {
    const engine = engineRef.current;
    if (!engine) return;
    setModelControls(model.controls);
    setModelInfo(
      `${model.name} · ${formatSize(model.byteLength)}` + (model.trainedAt ? ` · trained at ${model.trainedAt}px` : ''),
    );
    setSelectedModel(source);
    setConfigState(engine.getConfig());
  }, []);

  const loadListing = useCallback(
    async (listing: ModelListing) => {
      const engine = engineRef.current;
      if (!engine) return;
      setLoadingModel(true);
      setNotice(null);
      try {
        applyLoadedModel(await engine.loadModel(modelUrl(listing.file)), listing.file);
      } catch (error) {
        setNotice(error instanceof Error ? error.message : String(error));
      } finally {
        setLoadingModel(false);
      }
    },
    [applyLoadedModel],
  );

  const refreshSaved = useCallback(() => {
    void listSavedModels().then(setSaved);
  }, []);

  const loadSaved = useCallback(
    async (meta: SavedModelMeta) => {
      const engine = engineRef.current;
      if (!engine) return;
      setLoadingModel(true);
      setNotice(null);
      try {
        applyLoadedModel(engine.loadModelFromBuffer(await loadSavedModel(meta.id)), meta.id);
      } catch (error) {
        setNotice(error instanceof Error ? error.message : String(error));
      } finally {
        setLoadingModel(false);
      }
    },
    [applyLoadedModel],
  );

  const loadModelFile = useCallback(async (file: File) => {
    const engine = engineRef.current;
    if (!engine) return;
    setNotice(null);
    try {
      applyLoadedModel(engine.loadModelFromBuffer(await file.arrayBuffer()), file.name);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    }
  }, [applyLoadedModel]);

  /**
   * Applies settings read out of an image, and reloads the model they were made with when it is
   * one this build can reach. A model that is not available is reported rather than substituted —
   * the same sliders on a different network is not the same look, and silently pretending
   * otherwise would be worse than saying so.
   */
  const applyParameters = useCallback(
    async (parameters: EmbeddedParameters) => {
      const engine = engineRef.current;
      if (!engine) return;

      const wanted = parameters.model;
      if (wanted) {
        const listing = listings.find((entry) => entry.file === wanted.source);
        const savedMatch = saved.find((entry) => entry.id === wanted.source || entry.name === wanted.name);
        if (listing) await loadListing(listing);
        else if (savedMatch) await loadSaved(savedMatch);
        else {
          setNotice(
            `These settings were made with model "${wanted.name}", which is not in this build. ` +
              'Everything else has been applied; load that model to match exactly.',
          );
        }
      }

      // Applied after the model, because loading one resets the control vector to its defaults.
      engine.setConfig(parameters.config);
      setConfigState(engine.getConfig());
      engine.discardFeedback();
      setFoundParameters(null);
    },
    [listings, saved, loadListing, loadSaved],
  );

  /**
   * Measures every shipped model at the current capture size, on this machine.
   *
   * File size is a poor proxy for frame cost — a wide shallow network and a narrow deep one can
   * weigh the same and cost very different amounts — and the only number that settles it is the one
   * measured on the GPU that will actually run it.
   */
  const compareModels = useCallback(async (): Promise<ModelBenchmarkRow[]> => {
    const engine = engineRef.current;
    if (!engine) throw new Error('The engine is not running.');

    const wasRunning = engine.status().running;
    const restore = selectedModel;
    const rows: ModelBenchmarkRow[] = [];

    engine.stop();
    try {
      for (const listing of listings) {
        await engine.loadModel(modelUrl(listing.file));
        engine.setConfig({ processor: 'model' });
        engine.discardFeedback();
        const ms = engine.measure();
        rows.push({
          label: listing.name,
          detail: listing.width ? `${listing.width}×${listing.blocks} · ${formatSize(listing.bytes)}` : formatSize(listing.bytes),
          msPerFrame: ms,
          fps: ms > 0 ? 1000 / ms : 0,
        });
        await new Promise((resolve) => setTimeout(resolve, 30));
      }
    } finally {
      const original = listings.find((entry) => entry.file === restore);
      if (original) await loadListing(original);
      if (wasRunning) engine.start();
    }

    return rows;
  }, [listings, selectedModel, loadListing]);

  /**
   * Starts or stops microphone input.
   *
   * The permission prompt is the reason this is a button rather than something switched on by
   * default — asking for a microphone unprompted, on a page that is showing a camera, is not a
   * thing to do quietly.
   */
  const toggleAudio = useCallback(
    async (enabled: boolean) => {
      const engine = engineRef.current;
      if (!engine) return;

      if (!enabled) {
        engine.setAudio(null);
        setAudioLabel(null);
        patchConfig({ audio: { ...config.audio, enabled: false } });
        return;
      }

      try {
        const analyser = await AudioAnalyser.open();
        engine.setAudio(analyser);
        setAudioLabel(analyser.label);
        patchConfig({ audio: { ...config.audio, enabled: true } });
      } catch (error) {
        setNotice(error instanceof Error ? error.message : String(error));
        patchConfig({ audio: { ...config.audio, enabled: false } });
      }
    },
    [config.audio, patchConfig],
  );

  const setModulation = useCallback(
    (id: string, modulation: Modulation | null) => {
      const next = { ...config.modulations };
      if (modulation) next[id] = modulation;
      else delete next[id];
      patchConfig({ modulations: next });
    },
    [config.modulations, patchConfig],
  );

  /** The routing props every modulatable slider needs, so they are not repeated ten times. */
  const routed = (targetId: string) => ({
    targetId,
    modulations: config.modulations,
    onModulationChange: setModulation,
    audioEnabled: config.audio.enabled,
  });

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
        download(blob, timestampedName('dreamnet', extensionForMimeType(blob.type)));
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
        <h1>Waking Dream cannot start</h1>
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
    <div className="app" data-immersive={immersive ? 'true' : 'false'}>
      <main className="stage">
        <canvas ref={canvasRef} />

        <div className="stage-overlay">
          {sourceLabel !== 'none' && multipleCameras && engineRef.current?.currentSource?.kind === 'camera' ? (
            <button className="overlay-button" onClick={() => void flipCamera()} title="Switch between the front and rear cameras">
              ⟲ Flip
            </button>
          ) : null}
          <button
            className="overlay-button"
            onClick={toggleImmersive}
            title={immersive ? 'Show the controls' : 'Give the whole screen to the picture'}
          >
            {immersive ? '↙ Controls' : '⤢ Full screen'}
          </button>
        </div>
        {sourceLabel === 'none' ? (
          <div className="empty-stage">
            <h1>Waking Dream</h1>
            <p>Real-time dream filtering. Pick a source to begin.</p>
            <ButtonRow>
              <button className="button primary" onClick={() => void openCamera()}>
                Use camera
              </button>
              <FileButton label="Open video" accept="video/*" onFile={(file) => void attachSource(() => VideoFileSource.open(file))} />
              <FileButton label="Open image" accept="image/*" onFile={(file) => void openImage(file)} />
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
        {status?.error ? (
          <div className="stage-error">
            {status.error}
            {status.contextLost ? (
              <button className="button small" onClick={() => window.location.reload()}>Reload</button>
            ) : null}
          </div>
        ) : null}
      </main>

      <aside className="panel">
        <header className="panel-header">
          <h1>Waking Dream</h1>
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

        {foundParameters ? (
          <div className="notice found-parameters">
            <span>This image carries settings — {describeParameters(foundParameters)}</span>
            <ButtonRow>
              <button className="button small" onClick={() => void applyParameters(foundParameters)}>Apply</button>
              <button className="button small" onClick={() => setFoundParameters(null)}>Dismiss</button>
            </ButtonRow>
          </div>
        ) : null}

        <Section title="Source" hint={sourceLabel}>
          <ButtonRow>
            <button className="button" onClick={() => void openCamera(undefined, cameraFacing)}>Camera</button>
            {multipleCameras ? (
              <button className="button" onClick={() => void flipCamera()}>
                Flip to {cameraFacing === 'user' ? 'rear' : 'front'}
              </button>
            ) : null}
            <FileButton label="Video…" accept="video/*" onFile={(file) => void attachSource(() => VideoFileSource.open(file))} />
            <FileButton label="Image…" accept="image/*" onFile={(file) => void openImage(file)} />
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
              { value: 'model', label: 'DreamNet model', disabled: !modelInfo },
            ]}
            onChange={(processor) => patchConfig({ processor })}
          />
          {listings.length > 0 || saved.length > 0 ? (
            <Choice
              label={loadingModel ? 'Model (loading…)' : 'Model'}
              value={selectedModel}
              options={[
                ...(selectedModel && !listings.some((l) => l.file === selectedModel) && !saved.some((m) => m.id === selectedModel)
                  ? [{ value: selectedModel, label: `${selectedModel} (from disk)` }]
                  : []),
                ...listings.map((listing) => ({
                  value: listing.file,
                  label:
                    `${listing.name} · ${formatSize(listing.bytes)}` +
                    (costLabel(listing) ? ` · ${costLabel(listing)}` : ''),
                })),
                ...saved.map((meta) => ({
                  value: meta.id,
                  label: `${meta.name} · ${formatSize(meta.bytes)} · saved here`,
                })),
              ]}
              onChange={(value) => {
                const listing = listings.find((entry) => entry.file === value);
                if (listing) {
                  void loadListing(listing);
                  return;
                }
                const meta = saved.find((entry) => entry.id === value);
                if (meta) void loadSaved(meta);
              }}
            />
          ) : null}
          {saved.some((meta) => meta.id === selectedModel) ? (
            <ButtonRow>
              <button
                className="button small"
                onClick={() => {
                  const meta = saved.find((entry) => entry.id === selectedModel);
                  if (meta) void deleteSavedModel(meta.id).then(refreshSaved);
                }}
              >
                Delete saved model
              </button>
            </ButtonRow>
          ) : null}
          <ButtonRow>
            <FileButton label="Load .dnw model…" accept=".dnw" onFile={(file) => void loadModelFile(file)} />
          </ButtonRow>
          {modelInfo ? (
            <p className="note">
              {(() => {
                const listing = listings.find((entry) => entry.file === selectedModel);
                return listing?.width
                  ? `${listing.width} channels × ${listing.blocks} blocks · ${costLabel(listing)} · `
                  : '';
              })()}
              {modelInfo}
              {listings.find((entry) => entry.file === selectedModel)?.description
                ? ` — ${listings.find((entry) => entry.file === selectedModel)!.description}`
                : ''}
            </p>
          ) : (
            <p className="note">
              No trained model shipped with this build. Train one with <code>train/style.py</code> for a
              specific pattern or <code>train/train.py</code> to distill DeepDream, or stay on shallow
              ascent — it runs the same octaves and feedback with a hand-built one-layer filter bank.
            </p>
          )}
        </Section>

        <Section title="Capture size" hint={`${config.captureSize}px`}>
          <ButtonRow>
            {SIZE_PRESETS.filter((size) => size <= getDeviceLimits().maxCaptureSize).map((size) => (
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
            max={getDeviceLimits().maxCaptureSize}
            step={8}
            onChange={(captureSize) => patchConfig({ captureSize })}
            format={(value) => `${value} px`}
            title="The resolution the network actually sees. Cost grows with its square."
          />
          <BenchmarkPanel onRun={runBenchmark} onCompareModels={listings.length > 1 ? compareModels : undefined} />
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
            <ModulatedSlider
            {...routed('shallow.stepSize')}
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
            <ModulatedSlider
            {...routed('shallow.octaveScale')}
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
              label="Colour hold"
              value={config.shallow.colourHold}
              min={0}
              max={0.4}
              step={0.005}
              onChange={(colourHold) => patchConfig({ shallow: { ...config.shallow, colourHold } })}
              title="Pulls each ascent step back toward the source frame. Raise this if colours are running away to saturated primaries."
            />
            <Toggle
              label="Share gradient across colours"
              checked={config.shallow.sharedGradient}
              onChange={(sharedGradient) => patchConfig({ shallow: { ...config.shallow, sharedGradient } })}
              title="On, one scale is applied to R, G and B together, which keeps hue. Off, each channel is normalized alone — stronger colour shifts, and it will rail to primaries."
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

        {config.processor === 'model' && modelInfo ? (
          <Section
            title="Model scale"
            hint={config.modelOctaves > 1 ? `${config.modelOctaves} octaves` : 'single pass'}
            info="A model draws its motifs at a size fixed in its own input pixels, so a half-size frame makes everything it draws twice as large. That is the same lever --style-size pulls during training — what separates pandas from pandas-big — except here it costs a second pass instead of a second model. Each extra octave adds about a quarter of a frame at scale 2."
          >
            <Slider
              label="Octaves"
              value={config.modelOctaves}
              min={1}
              max={4}
              step={1}
              onChange={(modelOctaves) => patchConfig({ modelOctaves })}
              title="Runs the model at several resolutions and keeps the coarse structure from one and the fine detail from another. 1 is a single pass."
            />
            {config.modelOctaves > 1 ? (
              <ModulatedSlider
                {...routed('model.scaleMix')}
                label="Coarse structure"
                value={config.modelScaleMix}
                min={0}
                max={1}
                step={0.01}
                onChange={(modelScaleMix) => patchConfig({ modelScaleMix })}
                title="How much of the coarse pass's structure replaces the full-resolution pass's own. 0 is exactly the single pass."
              />
            ) : null}
            {config.modelOctaves > 1 ? (
              <Slider
                label="Octave scale"
                value={config.modelOctaveScale}
                min={1.4}
                max={3}
                step={0.1}
                onChange={(modelOctaveScale) => patchConfig({ modelOctaveScale })}
                title="How far apart the resolutions sit. Larger means the coarse pass draws much bigger motifs."
              />
            ) : null}
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

        <Section title="Train a style" hint="in this browser" defaultOpen={false} lazy>
          <TrainPanel
            getSource={() => engineRef.current?.currentSource ?? null}
            onUseModel={(buffer, name) => {
              const engine = engineRef.current;
              if (!engine) return;
              try {
                applyLoadedModel(engine.loadModelFromBuffer(buffer), name);
              } catch (error) {
                setNotice(error instanceof Error ? error.message : String(error));
              }
            }}
            onSavedModelsChanged={refreshSaved}
            onBusyChange={(training) => {
              // The runtime and the trainer would otherwise be competing for the GPU sixty times a
              // second, which makes the training run crawl and the preview stutter.
              const engine = engineRef.current;
              if (!engine) return;
              if (training) engine.stop();
              else engine.start();
            }}
          />
        </Section>

        <Section title="Sound" hint={config.audio.enabled ? audioLabel ?? 'on' : 'off'} defaultOpen={false}>
          <Toggle
            label="Drive the model from sound"
            checked={config.audio.enabled}
            onChange={(enabled) => void toggleAudio(enabled)}
            title="Splits the microphone into frequency bands and feeds each one to a model control."
          />

          {config.processor !== 'model' || modelControls.length === 0 ? (
            <p className="note">
              This drives a loaded model's controls, so it needs a model with sliders. Anything
              trained on several styles has one per style — the point being that different parts of
              the spectrum can push different textures forward.
            </p>
          ) : null}

          {config.audio.enabled ? (
            <>
              <div className="meters">
                {FREQUENCY_BANDS.map((band, index) => (
                  <div className="meter" key={band.name} title={`${band.low}–${band.high} Hz`}>
                    <div className="meter-track">
                      <span style={{ height: `${Math.round((status?.audioLevels[index] ?? 0) * 100)}%` }} />
                    </div>
                    <span className="meter-label">{band.label}</span>
                  </div>
                ))}
              </div>

              <Slider
                label="Amount"
                value={config.audio.amount}
                min={0}
                max={1}
                step={0.01}
                onChange={(amount) => patchConfig({ audio: { ...config.audio, amount } })}
                title="Crossfade between the slider positions and the sound. 1 is fully sound-driven."
              />
              <Slider
                label="Sensitivity"
                info="Every band is already normalized against its own recent peak, which is why they all sit mid-range on their own. Raise this to make them compete instead: whichever is loudest reads full and the others drop away behind it."
                value={config.audio.sensitivity}
                min={0}
                max={1}
                step={0.01}
                onChange={(sensitivity) => patchConfig({ audio: { ...config.audio, sensitivity } })}
                title="How hard the assigned bands compete. At 1 the loudest of them reads full and the others are pushed down a curve, so the gap between bass and voice is obvious rather than subtle."
              />
              <Slider
                label="Gain"
                value={config.audio.gain}
                min={0.2}
                max={4}
                step={0.05}
                onChange={(gain) => patchConfig({ audio: { ...config.audio, gain } })}
                title="Pushes quiet material up into range. Levels are already normalized per band against their own recent peak."
              />

              {modelControls.map((control, index) => (
                <Choice
                  key={control.name}
                  label={control.label}
                  value={String(config.audio.assignments[index] ?? index)}
                  options={FREQUENCY_BANDS.map((band, bandIndex) => ({
                    value: String(bandIndex),
                    label: `${band.label} · ${band.low}–${band.high} Hz`,
                  }))}
                  onChange={(value) => {
                    const assignments = config.audio.assignments.slice();
                    assignments[index] = Number(value);
                    patchConfig({ audio: { ...config.audio, assignments } });
                  }}
                />
              ))}
            </>
          ) : null}
        </Section>

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
          <ModulatedSlider
            {...routed('feedback.previous')}
            label="Previous output"
            value={config.feedback.previous}
            min={0}
            max={1}
            step={0.01}
            onChange={(previous) => patchConfig({ feedback: { ...config.feedback, previous } })}
          />
          <ModulatedSlider
            {...routed('feedback.zoom')}
            label="Zoom per frame"
            value={config.feedback.zoom}
            min={0.97}
            max={1.05}
            step={0.001}
            onChange={(zoom) => patchConfig({ feedback: { ...config.feedback, zoom } })}
            format={(value) => `${((value - 1) * 100).toFixed(1)}%`}
          />
          <ModulatedSlider
            {...routed('feedback.rotate')}
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
          <ModulatedSlider
            {...routed('feedback.fade')}
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

        <Section
          title="Colour"
          hint={config.colorPreservation > 0 ? `preserved ${config.colorPreservation.toFixed(2)}` : 'free'}
          info="Brightness is where the drawn structure lives, so it is never touched. At 1 the frame keeps the camera's colours exactly and every hallucinated form survives as light and shade within them. Applied before the feedback loop, so the recursion is held too."
        >
          <ModulatedSlider
            {...routed('colorPreservation')}
            label="Colour preservation"
            value={config.colorPreservation}
            min={0}
            max={1}
            step={0.01}
            onChange={(colorPreservation) => patchConfig({ colorPreservation })}
            title="Pulls hue and saturation back toward the source frame, leaving brightness — and so the drawn structure — alone. At 1 the camera's colours are kept exactly."
          />
        </Section>

        <Section title="Display" defaultOpen={false}>
          <ModulatedSlider
            {...routed('display.mix')}
            label="Effect amount"
            value={config.display.mix}
            min={0}
            max={1}
            step={0.01}
            onChange={(mix) => patchConfig({ display: { ...config.display, mix } })}
          />
          <ModulatedSlider
            {...routed('display.gain')}
            label="Gain"
            value={config.display.gain}
            min={0.4}
            max={2}
            step={0.01}
            onChange={(gain) => patchConfig({ display: { ...config.display, gain } })}
          />
          <Toggle
            label="Fill the screen"
            checked={config.fillScreen}
            onChange={(fillScreen) => patchConfig({ fillScreen })}
            title="Shape the capture like the display and centre-crop the source into it, instead of fitting the whole frame with bars at the sides. Turned on automatically in full screen."
          />
          <ModulatedSlider
            {...routed('display.saturation')}
            label="Saturation"
            value={config.display.saturation}
            min={0}
            max={2}
            step={0.01}
            onChange={(saturation) => patchConfig({ display: { ...config.display, saturation } })}
          />
        </Section>

        <Section
          title="Capture"
          defaultOpen={false}
          info="Saved frames are PNGs with the full settings written into them, so a frame is a way back to the look that made it — reopen one here, or send it to someone else. Recordings are H.264 MP4 wherever the browser can manage it, which is what QuickTime and Photos open."
        >
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
                if (!canvas) return;
                void saveCanvasFrame(
                  canvas,
                  timestampedName('dreamnet', 'png'),
                  buildParameters(
                    config,
                    modelInfo ? { source: selectedModel, name: modelInfo.split(' · ')[0] } : null,
                    status?.captureWidth ?? 0,
                    status?.captureHeight ?? 0,
                  ),
                );
              }}
            >
              Save frame
            </button>
            <FileButton
              label="Settings from image…"
              accept="image/png"
              onFile={(file) => {
                void readParametersFromFile(file).then((parameters) => {
                  if (parameters) void applyParameters(parameters);
                  else setNotice('That PNG carries no DreamNet settings.');
                });
              }}
            />
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
            {status ? (
              <>
                {status.programCount} shaders ·{' '}
                <span className={status.poolMegabytes > status.poolBudgetMegabytes * 0.85 ? 'slow' : undefined}>
                  {status.poolMegabytes.toFixed(0)}/{status.poolBudgetMegabytes.toFixed(0)} MB textures
                </span>
                {status.poolMegabytes > status.poolBudgetMegabytes * 0.85
                  ? ' — near the ceiling; drop capture size or octaves'
                  : ''}
              </>
            ) : null}
            {status && !status.supportsGpuTiming ? ' · no GPU timer' : ''}
          </div>
        </footer>
      </aside>
    </div>
  );
}
