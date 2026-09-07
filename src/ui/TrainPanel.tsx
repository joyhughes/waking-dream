import { useCallback, useEffect, useRef, useState } from 'react';
import { saveModel } from '../model/storage';
import { download, timestampedName } from '../pipeline/recorder';
import type { FrameSource } from '../pipeline/sources';
import type {
  FeatureNetworkId,
  FeatureNetworkOption,
  StyleInput,
  TrainingConfig,
  TrainingProgress,
  TrainingResult,
} from '../train';
import { ButtonRow, Choice, FileButton, Hint, Slider } from './Controls';

/**
 * Train a style model in the page.
 *
 * The whole training module is behind a dynamic import, loaded the first time this panel is opened,
 * because TensorFlow.js and a feature network are several megabytes that the viewing path has no
 * use for. Until then this component knows the types and nothing else.
 */

type TrainModule = typeof import('../train');

let modulePromise: Promise<TrainModule> | null = null;
function loadTrainModule(): Promise<TrainModule> {
  modulePromise ??= import('../train');
  return modulePromise;
}

interface TrainPanelProps {
  getSource: () => FrameSource | null;
  onUseModel: (buffer: ArrayBuffer, name: string) => void;
  onSavedModelsChanged: () => void;
  onBusyChange: (busy: boolean) => void;
}

export function TrainPanel({ getSource, onUseModel, onSavedModelsChanged, onBusyChange }: TrainPanelProps) {
  const [module, setModule] = useState<TrainModule | null>(null);
  const [config, setConfig] = useState<TrainingConfig | null>(null);
  const [networks, setNetworks] = useState<FeatureNetworkOption[]>([]);

  const [styles, setStyles] = useState<StyleInput[]>([]);
  const [frames, setFrames] = useState<HTMLCanvasElement[]>([]);
  const [modelName, setModelName] = useState('my-style');
  const [frameCount, setFrameCount] = useState(24);
  const [frameSpacing, setFrameSpacing] = useState(0.5);

  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [result, setResult] = useState<TrainingResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [savedId, setSavedId] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const previewRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    void loadTrainModule()
      .then((loaded) => {
        if (cancelled) return;
        setModule(loaded);
        setConfig(loaded.DEFAULT_TRAINING_CONFIG);
        setNetworks(loaded.FEATURE_NETWORKS);
      })
      .catch((caught) => !cancelled && setError(caught instanceof Error ? caught.message : String(caught)));
    return () => {
      cancelled = true;
    };
  }, []);

  // The preview arrives as ImageData rather than as a canvas, so that the trainer never touches
  // the DOM and can be tested without one.
  useEffect(() => {
    const canvas = previewRef.current;
    const preview = progress?.preview;
    if (!canvas || !preview) return;
    canvas.width = preview.width;
    canvas.height = preview.height;
    canvas.getContext('2d')?.putImageData(preview, 0, 0);
  }, [progress?.preview]);

  const addStyles = useCallback(
    async (files: File[]) => {
      const loaded = await loadTrainModule();
      const added: StyleInput[] = [];
      for (const file of files) {
        try {
          added.push({ name: file.name.replace(/\.[^.]+$/, ''), image: await loaded.loadImageFile(file) });
        } catch (caught) {
          setError(caught instanceof Error ? caught.message : String(caught));
        }
      }
      setStyles((previous) => [...previous, ...added]);
    },
    [],
  );

  const captureFrames = useCallback(async () => {
    const loaded = await loadTrainModule();
    const source = getSource();
    if (!source) {
      setError('Start a camera, video, or image source first — that is what the frames come from.');
      return;
    }
    setError(null);
    setBusy(true);
    try {
      const captured = await loaded.captureFromSource(source, frameCount, Math.round(frameSpacing * 1000));
      setFrames((previous) => [...previous, ...captured]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }, [getSource, frameCount, frameSpacing]);

  const addContentFiles = useCallback(async (files: File[]) => {
    const loaded = await loadTrainModule();
    const added: HTMLCanvasElement[] = [];
    for (const file of files) {
      try {
        const image = await loaded.loadImageFile(file);
        added.push(loaded.snapshot(image, image.naturalWidth, image.naturalHeight, 384));
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    }
    setFrames((previous) => [...previous, ...added]);
  }, []);

  /**
   * Keeps a finished model, in the browser.
   *
   * Called automatically the moment a run finishes rather than waiting for a button. A trained
   * model otherwise exists only in the live page, and a reload — which anything from a config
   * change to a stray refresh can cause — silently throws away however many minutes went into it.
   * Losing work to an unclicked button is not a tradeoff worth making for a little tidiness.
   */
  const saveToBrowser = useCallback(
    async (trained: TrainingResult) => {
      try {
        const meta = await saveModel(
          {
            name: trained.name,
            description: `Trained in the browser on ${styles.map((style) => style.name).join(', ') || 'no styles'}.`,
            bytes: trained.bytes,
            trainedAt: config?.cropSize,
            controls: trained.controls.map((control) => control.label),
          },
          trained.buffer,
        );
        setSavedId(meta.id);
        onSavedModelsChanged();
        return true;
      } catch (caught) {
        // Private browsing and locked-down profiles refuse IndexedDB. The model is still in the
        // page and still downloadable, so this is a warning rather than a failure.
        setError(
          `Trained fine, but this browser would not store the model (${
            caught instanceof Error ? caught.message : String(caught)
          }). Download it before reloading.`,
        );
        return false;
      }
    },
    [styles, config, onSavedModelsChanged],
  );

  const start = useCallback(async () => {
    const loaded = module ?? (await loadTrainModule());
    if (!config) return;

    setError(null);
    setResult(null);
    setSavedId(null);
    setBusy(true);
    onBusyChange(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const trained = await loaded.trainStyleModel({
        styles,
        contentFrames: frames,
        modelName,
        config,
        onProgress: setProgress,
        signal: controller.signal,
      });
      setResult(trained);
      // Straight into the fast runtime, so the thing just trained is what is on screen.
      onUseModel(trained.buffer, trained.name);
      // And kept, before anything else can go wrong.
      await saveToBrowser(trained);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      abortRef.current = null;
      setBusy(false);
      onBusyChange(false);
    }
  }, [module, config, styles, frames, modelName, onUseModel, onBusyChange, saveToBrowser]);

  if (error && !module) {
    return <p className="error">{error}</p>;
  }

  if (!config) {
    return <p className="note">Loading the trainer…</p>;
  }

  const patch = (values: Partial<TrainingConfig>) => setConfig({ ...config, ...values });
  const ready = styles.length > 0 && frames.length > 0;
  const percent = progress && progress.total > 0 ? (progress.step / progress.total) * 100 : 0;
  const remainingMs =
    progress && progress.step > 0 && progress.phase === 'training'
      ? (progress.elapsedMs / progress.step) * (progress.total - progress.step)
      : 0;

  return (
    <>
      {error ? <p className="error">{error}</p> : null}

      <ButtonRow>
        <FileButton label="Style images…" accept="image/*" multiple onFiles={(files) => void addStyles(files)} />
        <Hint text="Pick images with texture all over rather than an interesting composition — the network learns which strokes and colours go together and throws away where they were. Each image becomes its own slider, and the sliders blend." />
        {styles.length > 0 ? (
          <button className="button small" onClick={() => setStyles([])} disabled={busy}>Clear</button>
        ) : null}
      </ButtonRow>
      {styles.length > 0 ? (
        <p className="note">{styles.length} style{styles.length === 1 ? '' : 's'}: {styles.map((style) => style.name).join(', ')} — each gets its own slider, and they blend.</p>
      ) : null}

      <Slider
        label="Frames to capture"
        value={frameCount}
        min={4}
        max={120}
        step={4}
        onChange={setFrameCount}
      />
      <Slider
        label="Seconds between frames"
        value={frameSpacing}
        min={0.1}
        max={3}
        step={0.1}
        onChange={setFrameSpacing}
        format={(value) => `${value.toFixed(1)}s`}
        title="Spread the grabs out so the frames are genuinely different scenes rather than one moment sampled repeatedly."
      />
      <ButtonRow>
        <button className="button" onClick={() => void captureFrames()} disabled={busy}>
          {busy ? 'Capturing…' : `Capture ${frameCount} frames (${Math.round(frameCount * frameSpacing)}s)`}
        </button>
        <FileButton label="Add photos…" accept="image/*" multiple onFiles={(files) => void addContentFiles(files)} />
        <Hint text="Content frames are only things the network must keep recognisable while repainting them — the look comes entirely from the style images. Capturing from your own camera means it sees the lighting it will actually run in. Three to five hundred varied frames is the comfortable range." />
        {frames.length > 0 ? (
          <button className="button small" onClick={() => setFrames([])} disabled={busy}>Clear</button>
        ) : null}
      </ButtonRow>
      {frames.length > 0 ? <p className="note">{frames.length} content frames.</p> : null}

      <Choice<FeatureNetworkId>
        label="Feature network"
        info={networks.map((network) => `${network.label}: ${network.description}`).join(' ')}
        value={config.featureNetwork}
        options={networks.map((network) => ({
          value: network.id,
          label: `${network.label} · ${network.downloadLabel}`,
        }))}
        onChange={(featureNetwork) => patch({ featureNetwork })}
      />


      <Slider
        label="Iterations"
        value={config.iterations}
        min={100}
        max={4000}
        step={50}
        onChange={(iterations) => patch({ iterations })}
        title="More is better and slower. You can stop early and keep what it has."
      />
      <Slider
        label="Pattern scale"
        value={config.styleSize}
        min={96}
        max={768}
        step={16}
        onChange={(styleSize) => patch({ styleSize })}
        format={(value) => `${value} px`}
        title="The size the style image is read at. Smaller gives finer, denser motifs; larger gives coarser, bolder ones. The most consequential control here."
      />
      <Slider
        label="Style weight"
        value={config.styleWeight}
        min={0.5}
        max={60}
        step={0.5}
        onChange={(styleWeight) => patch({ styleWeight })}
        title="Raise it if the output still looks like the photo; lower it if the photo has vanished into wallpaper."
      />
      <Slider
        label="Crop size"
        value={config.cropSize}
        min={96}
        max={256}
        step={16}
        onChange={(cropSize) => patch({ cropSize })}
        format={(value) => `${value} px`}
      />
      <Slider
        label="Network width"
        value={config.width}
        min={4}
        max={24}
        step={2}
        onChange={(width) => patch({ width })}
        title="The size/speed tradeoff of the trained model itself, and of the training run."
      />
      <Slider
        label="Residual blocks"
        value={config.blocks}
        min={1}
        max={6}
        step={1}
        onChange={(blocks) => patch({ blocks })}
      />
      <Slider
        label="Temporal stability"
        value={config.warpWeight}
        min={0}
        max={1}
        step={0.05}
        onChange={(warpWeight) => patch({ warpWeight })}
        title="Asks the network to move its detail with the picture. Costs an extra pass per step; set to 0 for stills."
      />

      <label className="control">
        <span className="control-label">Model name</span>
        <input
          className="text-input"
          value={modelName}
          onChange={(event) => setModelName(event.target.value.replace(/[^a-zA-Z0-9._-]/g, '-'))}
        />
      </label>

      <ButtonRow>
        {busy && abortRef.current ? (
          <button className="button recording" onClick={() => abortRef.current?.abort()}>
            Stop and keep
          </button>
        ) : (
          <button className="button primary" onClick={() => void start()} disabled={!ready || busy}>
            Train
          </button>
        )}
      </ButtonRow>
      {!ready ? <p className="note">Add at least one style image and some content frames.</p> : null}

      {progress ? (
        <div className="training-progress">
          <div className="progress-bar"><span style={{ width: `${percent}%` }} /></div>
          <p className="note">
            {progress.phase === 'training'
              ? `step ${progress.step} / ${progress.total}` +
                (remainingMs > 0 ? ` · about ${formatDuration(remainingMs)} left` : '')
              : progress.message ?? progress.phase}
          </p>
          {progress.phase === 'training' ? (
            <p className="note dim">
              content {progress.losses.content.toFixed(4)} · style {progress.losses.style.toFixed(4)} ·
              tv {progress.losses.tv.toFixed(5)} · warp {progress.losses.warp.toFixed(5)}
            </p>
          ) : null}
          <canvas ref={previewRef} className="training-preview" />
        </div>
      ) : null}

      {result ? (
        <>
          <p className="note">
            Done — {result.steps} steps in {formatDuration(result.elapsedMs)}, {(result.bytes / 1024).toFixed(0)} kB.
            It is running on the canvas{savedId ? ' and saved to this browser, so a reload will not lose it' : ''}.
          </p>
          <ButtonRow>
            {savedId ? null : (
              <button className="button" onClick={() => void saveToBrowser(result)}>Save to this browser</button>
            )}
            <button
              className="button"
              onClick={() => download(new Blob([result.buffer]), `${result.name || timestampedName('dreamnet', 'dnw')}.dnw`)}
            >
              Download .dnw
            </button>
          </ButtonRow>
          <p className="note">
            The downloaded file is the whole model.
            <Hint text="Send it to someone and they can open it with Load .dnw model…; drop it in public/models/ and run pnpm models:index to ship it with the deployed build." />
          </p>
        </>
      ) : null}
    </>
  );
}

function formatDuration(ms: number): string {
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, '0')}s`;
}
