/**
 * Where frames come from: the camera, a video file, or a single still image.
 *
 * All three end up as something `texImage2D` accepts, so the rest of the pipeline never branches on
 * which one is active. The still image is not a degenerate case — with feedback on, a photograph
 * fed back into itself is the mode that produces the classic evolving dream, and it is the one
 * source where every frame's input is the previous frame's output rather than new data.
 */

export type SourceKind = 'camera' | 'video' | 'image';

export interface FrameSource {
  readonly kind: SourceKind;
  /** What to upload. Null while the source is still loading or has nothing decoded yet. */
  readonly frame: TexImageSource | null;
  readonly width: number;
  readonly height: number;
  /** Cameras are mirrored by default, because an unmirrored self-view is disorienting. */
  readonly defaultMirror: boolean;
  readonly label: string;
  dispose(): void;
}

/** Which way a phone camera points. Desktops have one camera and report it as `user`. */
export type CameraFacing = 'user' | 'environment';

export class CameraSource implements FrameSource {
  readonly kind = 'camera';
  readonly label: string;
  /** Which camera this actually is, so the flip control knows what to ask for next. */
  readonly facing: CameraFacing;

  private constructor(
    private readonly video: HTMLVideoElement,
    private readonly stream: MediaStream,
    label: string,
    facing: CameraFacing,
  ) {
    this.label = label;
    this.facing = facing;
  }

  /**
   * The front camera is mirrored and the rear one is not.
   *
   * A self-view that is not mirrored is disorienting — moving left sends your reflection right. A
   * rear camera is not a reflection at all, and mirroring it would be simply wrong.
   */
  get defaultMirror(): boolean {
    return this.facing === 'user';
  }

  static async open(deviceId?: string, facing: CameraFacing = 'user', requestedHeight = 720): Promise<CameraSource> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser will not give a page camera access.');
    }

    // The camera is asked for more resolution than the capture size will use. Downscaling a sharp
    // frame on the GPU is free and looks better than asking the camera for a small frame, which on
    // most hardware means a cropped sensor readout rather than a scaled one.
    //
    // `facingMode` is how a phone camera is selected: device ids are unstable across sessions on
    // iOS and their labels are empty until permission has been granted at least once, so a device
    // picker cannot be built before the first successful open. Asking by which way it points works
    // on the very first call.
    const stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId
        ? { deviceId: { exact: deviceId }, height: { ideal: requestedHeight } }
        : { facingMode: { ideal: facing }, height: { ideal: requestedHeight } },
      audio: false,
    });

    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
    video.srcObject = stream;

    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error('The camera stream failed to start.'));
    });
    await video.play();

    const track = stream.getVideoTracks()[0];
    const label = track?.label || 'Camera';
    // What was asked for is not always what was given — a device with only one camera hands back
    // whatever it has — so the mirroring follows what the track reports rather than the request.
    const settings = track?.getSettings?.() as { facingMode?: string } | undefined;
    const actual: CameraFacing = settings?.facingMode === 'environment' ? 'environment' : facing;

    return new CameraSource(video, stream, label, actual);
  }

  /** Whether this device has more than one camera to flip between. */
  static async hasMultipleCameras(): Promise<boolean> {
    return (await CameraSource.listCameras()).length > 1;
  }

  static async listCameras(): Promise<MediaDeviceInfo[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((device) => device.kind === 'videoinput');
  }

  get frame(): TexImageSource | null {
    return this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA ? this.video : null;
  }

  get width(): number {
    return this.video.videoWidth || 1;
  }

  get height(): number {
    return this.video.videoHeight || 1;
  }

  dispose(): void {
    for (const track of this.stream.getTracks()) track.stop();
    this.video.srcObject = null;
  }
}

export class VideoFileSource implements FrameSource {
  readonly kind = 'video';
  readonly defaultMirror = false;
  readonly label: string;

  private constructor(
    /** Exposed so the transport UI can read position and rate directly rather than mirror them. */
    readonly video: HTMLVideoElement,
    private readonly objectUrl: string,
    label: string,
  ) {
    this.label = label;
  }

  static async open(file: File): Promise<VideoFileSource> {
    const objectUrl = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.loop = true;
    video.preload = 'auto';
    video.src = objectUrl;

    try {
      await new Promise<void>((resolve, reject) => {
        // `loadedmetadata` only promises dimensions and duration; it does not promise a decoded
        // frame, and uploading before one exists gives a black first second.
        video.onloadeddata = () => resolve();
        video.onerror = () => reject(new Error(`Could not decode "${file.name}".`));
      });
    } catch (error) {
      URL.revokeObjectURL(objectUrl);
      throw error;
    }

    await video.play().catch(() => {
      // Autoplay can be refused before the user has interacted with the page. The frame at time
      // zero is already decoded, so the pipeline runs on a still until they press play.
    });

    return new VideoFileSource(video, objectUrl, file.name);
  }

  get frame(): TexImageSource | null {
    return this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA ? this.video : null;
  }

  get width(): number {
    return this.video.videoWidth || 1;
  }

  get height(): number {
    return this.video.videoHeight || 1;
  }

  get playing(): boolean {
    return !this.video.paused;
  }

  togglePlay(): void {
    if (this.video.paused) void this.video.play();
    else this.video.pause();
  }

  seek(seconds: number): void {
    this.video.currentTime = Math.max(0, Math.min(this.video.duration || 0, seconds));
  }

  dispose(): void {
    this.video.pause();
    this.video.removeAttribute('src');
    this.video.load();
    URL.revokeObjectURL(this.objectUrl);
  }
}

export class ImageSource implements FrameSource {
  readonly kind = 'image';
  readonly defaultMirror = false;
  readonly label: string;

  private constructor(
    private readonly image: HTMLImageElement,
    private readonly objectUrl: string | null,
    label: string,
  ) {
    this.label = label;
  }

  static async open(file: File): Promise<ImageSource> {
    const objectUrl = URL.createObjectURL(file);
    try {
      const image = await loadImage(objectUrl);
      return new ImageSource(image, objectUrl, file.name);
    } catch (error) {
      URL.revokeObjectURL(objectUrl);
      throw error;
    }
  }

  static async fromUrl(url: string, label: string): Promise<ImageSource> {
    return new ImageSource(await loadImage(url), null, label);
  }

  /** Wraps a canvas the page drew itself, which is how the built-in test pattern gets in. */
  static async fromCanvas(canvas: HTMLCanvasElement, label: string): Promise<ImageSource> {
    return new ImageSource(await loadImage(canvas.toDataURL('image/png')), null, label);
  }

  get frame(): TexImageSource | null {
    return this.image.complete ? this.image : null;
  }

  get width(): number {
    return this.image.naturalWidth || 1;
  }

  get height(): number {
    return this.image.naturalHeight || 1;
  }

  dispose(): void {
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
  }
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Could not decode that image.'));
    image.src = src;
  });
}
