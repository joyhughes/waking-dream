import { useEffect, useState } from 'react';
import type { VideoFileSource } from '../pipeline/sources';
import { ButtonRow, Slider, Toggle } from './Controls';

/**
 * Transport for a video file source.
 *
 * Scrubbing matters more here than in an ordinary player: with feedback running, the effect depends
 * on how long it has been fed a given stretch of footage, so being able to sit on one moment,
 * step back, or slow the whole thing down is how a look gets found. Playback rate is exposed for
 * the same reason — at quarter speed the recursion gets four times as many frames to build on.
 */
export function VideoTransport({ source }: { source: VideoFileSource }) {
  const [, forceUpdate] = useState(0);
  const video = source.video;

  useEffect(() => {
    const id = window.setInterval(() => forceUpdate((n) => n + 1), 200);
    return () => window.clearInterval(id);
  }, []);

  const duration = Number.isFinite(video.duration) ? video.duration : 0;

  return (
    <>
      <ButtonRow>
        <button className="button" onClick={() => source.togglePlay()}>
          {source.playing ? 'Pause' : 'Play'}
        </button>
        <button className="button small" onClick={() => source.seek(video.currentTime - 1)}>−1s</button>
        <button className="button small" onClick={() => source.seek(video.currentTime + 1)}>+1s</button>
        <button className="button small" onClick={() => source.seek(0)}>Start</button>
      </ButtonRow>
      <Slider
        label="Position"
        value={Math.min(video.currentTime, duration)}
        min={0}
        max={Math.max(duration, 0.1)}
        step={0.05}
        onChange={(seconds) => source.seek(seconds)}
        format={(seconds) => `${seconds.toFixed(1)} / ${duration.toFixed(1)}s`}
      />
      <Slider
        label="Playback rate"
        value={video.playbackRate}
        min={0.1}
        max={2}
        step={0.05}
        onChange={(rate) => {
          video.playbackRate = rate;
          forceUpdate((n) => n + 1);
        }}
        format={(rate) => `${rate.toFixed(2)}×`}
        title="Slower playback gives the feedback loop more frames per second of footage to build on."
      />
      <Toggle
        label="Loop"
        checked={video.loop}
        onChange={(loop) => {
          video.loop = loop;
          forceUpdate((n) => n + 1);
        }}
      />
    </>
  );
}
