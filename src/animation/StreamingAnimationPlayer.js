/**
 * StreamingAnimationPlayer — real-time animation streaming via WebSocket or HTTP.
 *
 * Binary frame protocol (little-endian):
 *
 *   Header — 12 bytes:
 *     [0..3]   uint32  serverTimestampMs  — server wall-clock time for this frame
 *     [4..7]   uint32  frameId            — monotonically increasing frame counter
 *     [8]      uint8   jointCount         — number of joints encoded
 *     [9]      uint8   facsCount          — number of FACS weights encoded (≤ 52)
 *     [10..11] uint16  flags              — reserved, must be 0
 *
 *   Joint data — jointCount × 14 bytes each:
 *     [0..5]   3 × int16  position   (metres × 1 000 — i.e. millimetre precision)
 *     [6..13]  4 × int16  quaternion (normalized; each component × 32 767)
 *
 *   FACS data — facsCount × 2 bytes each:
 *     int16   morph weight (× 32 767; clamped 0..1 = 0..32767)
 *
 * Jitter buffer:
 *   Incoming frames are timestamped with the *local* arrival time.
 *   A configurable _targetDelayMs (default 60 ms) is added to form a
 *   "playback clock" that runs slightly behind the live edge.
 *   getInterpolatedPose(nowMs) returns a pose blended between the two
 *   frames that bracket the playback position.  When the buffer is
 *   exhausted (packet loss / stall) the last good pose is returned
 *   (hold/extrapolate) to avoid visual pops.
 *
 * Zero runtime dependencies.
 */

export const STREAM_TYPE_WS   = 'ws';
export const STREAM_TYPE_HTTP = 'http';

/** Maximum frames held in the jitter buffer before oldest are evicted. */
const BUFFER_MAX_FRAMES = 64;

/** Quantisation scales matching the binary protocol. */
const POS_SCALE  = 1 / 1000;   // int16 → metres
const NORM_SCALE = 1 / 32767;  // int16 → normalized float (joints + FACS)

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Decode a binary frame ArrayBuffer into a frame object. */
function decodeFrame(buffer) {
  if (buffer.byteLength < 12) {
    throw new RangeError(`[StreamingAnimationPlayer] Frame too short: ${buffer.byteLength} bytes`);
  }
  const view = new DataView(buffer);
  const serverTs   = view.getUint32(0, true);
  const frameId    = view.getUint32(4, true);
  const jointCount = view.getUint8(8);
  const facsCount  = view.getUint8(9);
  // flags at [10..11] reserved

  const minLength = 12 + jointCount * 14 + facsCount * 2;
  if (buffer.byteLength < minLength) {
    throw new RangeError(
      `[StreamingAnimationPlayer] Frame data too short: expected ≥${minLength}, got ${buffer.byteLength}`
    );
  }

  // Decode joint translations & rotations
  const joints = new Float32Array(jointCount * 7); // tx,ty,tz, qx,qy,qz,qw per joint
  let offset = 12;
  for (let j = 0; j < jointCount; j++) {
    const base = j * 7;
    joints[base]     = view.getInt16(offset,      true) * POS_SCALE;
    joints[base + 1] = view.getInt16(offset + 2,  true) * POS_SCALE;
    joints[base + 2] = view.getInt16(offset + 4,  true) * POS_SCALE;
    joints[base + 3] = view.getInt16(offset + 6,  true) * NORM_SCALE;
    joints[base + 4] = view.getInt16(offset + 8,  true) * NORM_SCALE;
    joints[base + 5] = view.getInt16(offset + 10, true) * NORM_SCALE;
    joints[base + 6] = view.getInt16(offset + 12, true) * NORM_SCALE;
    offset += 14;
  }

  // Decode FACS weights
  const facs = new Float32Array(facsCount);
  for (let f = 0; f < facsCount; f++) {
    facs[f] = Math.max(0, view.getInt16(offset, true) * NORM_SCALE);
    offset += 2;
  }

  return { serverTs, frameId, joints, facs, arrivalTs: performance.now() };
}

/** Encode a frame to binary (useful for mock server / testing). */
export function encodeFrame(frame) {
  const { serverTs, frameId, joints, facs } = frame;
  const jointCount = joints.length / 7 | 0;
  const facsCount  = facs.length;
  const buf  = new ArrayBuffer(12 + jointCount * 14 + facsCount * 2);
  const view = new DataView(buf);

  view.setUint32(0, serverTs >>> 0, true);
  view.setUint32(4, frameId  >>> 0, true);
  view.setUint8(8, jointCount);
  view.setUint8(9, facsCount);
  view.setUint16(10, 0, true); // flags

  let offset = 12;
  for (let j = 0; j < jointCount; j++) {
    const base = j * 7;
    view.setInt16(offset,      Math.round(joints[base]     / POS_SCALE),  true);
    view.setInt16(offset + 2,  Math.round(joints[base + 1] / POS_SCALE),  true);
    view.setInt16(offset + 4,  Math.round(joints[base + 2] / POS_SCALE),  true);
    view.setInt16(offset + 6,  Math.round(joints[base + 3] / NORM_SCALE), true);
    view.setInt16(offset + 8,  Math.round(joints[base + 4] / NORM_SCALE), true);
    view.setInt16(offset + 10, Math.round(joints[base + 5] / NORM_SCALE), true);
    view.setInt16(offset + 12, Math.round(joints[base + 6] / NORM_SCALE), true);
    offset += 14;
  }
  for (let f = 0; f < facsCount; f++) {
    view.setInt16(offset, Math.round(Math.max(0, Math.min(1, facs[f])) * 32767), true);
    offset += 2;
  }
  return buf;
}

/** Linear interpolation between two float arrays of equal length. */
function lerpArray(a, b, t, out) {
  for (let i = 0; i < a.length; i++) {
    out[i] = a[i] + (b[i] - a[i]) * t;
  }
}

/** Spherical linear interpolation for a quaternion stored at offset in a Float32Array. */
function slerpJointQuat(a, b, t, out, base) {
  let ax = a[base + 3], ay = a[base + 4], az = a[base + 5], aw = a[base + 6];
  let bx = b[base + 3], by = b[base + 4], bz = b[base + 5], bw = b[base + 6];

  let dot = ax * bx + ay * by + az * bz + aw * bw;
  if (dot < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; dot = -dot; }

  let s0, s1;
  if (dot > 0.9995) {
    // Quaternions nearly identical — use linear blend + renormalize
    s0 = 1 - t; s1 = t;
  } else {
    const angle  = Math.acos(Math.min(dot, 1));
    const sinInv = 1 / Math.sin(angle);
    s0 = Math.sin((1 - t) * angle) * sinInv;
    s1 = Math.sin(t * angle) * sinInv;
  }

  out[base + 3] = ax * s0 + bx * s1;
  out[base + 4] = ay * s0 + by * s1;
  out[base + 5] = az * s0 + bz * s1;
  out[base + 6] = aw * s0 + bw * s1;

  // Normalize to handle floating-point drift
  const len = Math.sqrt(
    out[base + 3] ** 2 + out[base + 4] ** 2 + out[base + 5] ** 2 + out[base + 6] ** 2
  );
  if (len > 1e-6) {
    const inv = 1 / len;
    out[base + 3] *= inv; out[base + 4] *= inv;
    out[base + 5] *= inv; out[base + 6] *= inv;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamingAnimationPlayer
// ─────────────────────────────────────────────────────────────────────────────

export class StreamingAnimationPlayer {
  /**
   * @param {object}  [opts]
   * @param {number}  [opts.targetDelayMs=60]    — jitter buffer target delay in ms
   * @param {number}  [opts.maxBufferFrames=64]  — max frames held before oldest are dropped
   * @param {number}  [opts.reconnectDelayMs=2000] — ms between auto-reconnect attempts
   * @param {boolean} [opts.autoReconnect=true]
   * @param {number}  [opts.smoothingAlpha=0.15] — exponential smoothing for latency estimate
   */
  constructor(opts = {}) {
    this._targetDelayMs    = opts.targetDelayMs    ?? 60;
    this._maxBufferFrames  = opts.maxBufferFrames  ?? BUFFER_MAX_FRAMES;
    this._reconnectDelayMs = opts.reconnectDelayMs ?? 2000;
    this._autoReconnect    = opts.autoReconnect    ?? true;
    this._smoothingAlpha   = opts.smoothingAlpha   ?? 0.15;

    /** @type {Array<{serverTs:number, frameId:number, joints:Float32Array, facs:Float32Array, arrivalTs:number}>} */
    this._buffer    = [];
    this._lastFrame = null;        // last consumed frame (for extrapolation)
    this._events    = {};

    this._ws            = null;
    this._url           = null;
    this._streamType    = STREAM_TYPE_WS;
    this._connected     = false;
    this._reconnectTimer= null;

    // HTTP streaming
    this._httpAbort     = null;

    // Stats
    this._totalFrames  = 0;
    this._droppedFrames= 0;
    this._fpsCounter   = 0;
    this._fpsTs        = performance.now();
    this._fps          = 0;
    this._smoothLatency= 0;

    // Playback start offset (localTs - targetDelay = playback clock)
    this._startLocalTs = null;
    this._startSrvTs   = null;
  }

  // ──────────────────────────────────────────────────────────────── connection

  /**
   * Connect to a streaming endpoint.
   * @param {string} url             — ws:// wss:// or http:// https:// URL
   * @param {'ws'|'http'} [type]     — override transport; auto-detected from URL scheme if omitted
   */
  connect(url, type) {
    this.disconnect();

    this._url = url;
    this._streamType = type
      ?? (url.startsWith('ws') ? STREAM_TYPE_WS : STREAM_TYPE_HTTP);

    if (this._streamType === STREAM_TYPE_WS) {
      this._connectWS(url);
    } else {
      this._connectHTTP(url);
    }
  }

  disconnect() {
    this._cancelReconnect();

    if (this._ws) {
      this._ws.onclose   = null;
      this._ws.onerror   = null;
      this._ws.onmessage = null;
      this._ws.onopen    = null;
      if (this._ws.readyState < 2) this._ws.close();
      this._ws = null;
    }

    if (this._httpAbort) {
      this._httpAbort.abort();
      this._httpAbort = null;
    }

    if (this._connected) {
      this._connected = false;
      this._emit('disconnect', { url: this._url });
    }
  }

  // ──────────────────────────────────────────────────────────────── public API

  /**
   * Return the interpolated/extrapolated pose for the given local timestamp.
   *
   * @param {number} [nowMs=performance.now()]
   * @returns {{
   *   joints:   Float32Array|null,  — jointCount × 7 floats (tx,ty,tz, qx,qy,qz,qw)
   *   facs:     Float32Array|null,  — facsCount floats [0..1]
   *   frameId:  number,
   *   age:      number,             — ms since this frame arrived
   *   interpolated: boolean
   * }|null}  null when no data has been received yet
   */
  getInterpolatedPose(nowMs = performance.now()) {
    if (this._buffer.length === 0 && !this._lastFrame) return null;

    // Compute playback position on the server timeline
    let playTs;
    if (this._startLocalTs !== null && this._startSrvTs !== null) {
      const localElapsed = nowMs - this._startLocalTs;
      playTs = this._startSrvTs + localElapsed - this._targetDelayMs;
    } else if (this._lastFrame) {
      return this._makeResult(this._lastFrame, false);
    } else {
      return null;
    }

    // Find bracketing frames
    const buf = this._buffer;

    if (buf.length === 0) {
      // Extrapolate: no buffered frame — return last
      return this._makeResult(this._lastFrame, false);
    }

    // Find first frame at or after playTs
    let hi = -1;
    for (let i = 0; i < buf.length; i++) {
      if (buf[i].serverTs >= playTs) { hi = i; break; }
    }

    if (hi === -1) {
      // All frames are in the past — consume the newest and extrapolate
      const newest = buf[buf.length - 1];
      this._lastFrame = newest;
      this._buffer = [];
      return this._makeResult(newest, false);
    }

    if (hi === 0) {
      // Play position is before the oldest buffered frame
      const oldest = buf[0];
      if (this._lastFrame) {
        // Interpolate between lastFrame and oldest
        const lo   = this._lastFrame;
        const span = oldest.serverTs - lo.serverTs;
        const t    = span > 0 ? Math.max(0, Math.min(1, (playTs - lo.serverTs) / span)) : 0;
        return this._interpolate(lo, oldest, t);
      }
      return this._makeResult(oldest, false);
    }

    // Normal case: interpolate between buf[hi-1] and buf[hi]
    const lo  = buf[hi - 1];
    const hi_ = buf[hi];
    const span = hi_.serverTs - lo.serverTs;
    const t    = span > 0 ? Math.max(0, Math.min(1, (playTs - lo.serverTs) / span)) : 0;

    // Evict frames that are now behind the playback pointer
    this._lastFrame = lo;
    this._buffer    = buf.slice(hi - 1);  // keep lo and everything after

    return this._interpolate(lo, hi_, t);
  }

  /**
   * Register an event listener.
   * @param {'connect'|'disconnect'|'data'|'error'|'drop'|'reconnecting'} event
   * @param {Function} cb
   */
  on(event, cb) {
    if (!this._events[event]) this._events[event] = [];
    this._events[event].push(cb);
    return this;
  }

  /** Remove a previously registered event listener. */
  off(event, cb) {
    if (!this._events[event]) return this;
    this._events[event] = this._events[event].filter(fn => fn !== cb);
    return this;
  }

  /** Performance statistics snapshot. */
  get stats() {
    return {
      connected:   this._connected,
      bufferSize:  this._buffer.length,
      totalFrames: this._totalFrames,
      dropped:     this._droppedFrames,
      fps:         this._fps,
      latencyMs:   Math.round(this._smoothLatency),
      targetDelayMs: this._targetDelayMs,
    };
  }

  /** Set the jitter buffer target delay in milliseconds. */
  setTargetDelay(ms) {
    this._targetDelayMs = Math.max(0, ms);
  }

  /**
   * Set the exponential smoothing factor for the latency estimate.
   * @param {number} alpha — value in (0, 1]; higher = faster tracking, lower = smoother
   */
  setSmoothingAlpha(alpha) {
    this._smoothingAlpha = Math.max(0.001, Math.min(1, alpha));
  }

  /**
   * Inject a pre-encoded binary frame directly into the jitter buffer.
   * Useful for testing, mock data sources, and in-page simulations without
   * a real network connection.
   * @param {ArrayBuffer} buffer — frame encoded with encodeFrame()
   */
  injectFrame(buffer) {
    this._receiveFrame(buffer);
  }

  destroy() {
    this.disconnect();
    this._buffer    = [];
    this._lastFrame = null;
    this._events    = {};
  }

  // ──────────────────────────────────────────────────────────────── private

  /** @private */
  _connectWS(url) {
    let ws;
    try {
      ws = new WebSocket(url);
    } catch (e) {
      this._emit('error', { message: e.message });
      this._scheduleReconnect();
      return;
    }
    ws.binaryType = 'arraybuffer';
    this._ws = ws;

    ws.onopen = () => {
      this._connected = true;
      this._resetPlaybackClock();
      this._emit('connect', { url, transport: 'ws' });
    };

    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        this._receiveFrame(ev.data);
      }
    };

    ws.onerror = (ev) => {
      this._emit('error', { message: 'WebSocket error', event: ev });
    };

    ws.onclose = (ev) => {
      const wasConnected = this._connected;
      this._connected = false;
      this._ws = null;
      if (wasConnected) this._emit('disconnect', { url, code: ev.code, reason: ev.reason });
      if (this._autoReconnect) this._scheduleReconnect();
    };
  }

  /** @private */
  _connectHTTP(url) {
    const ctrl = new AbortController();
    this._httpAbort = ctrl;

    const doFetch = async () => {
      try {
        const res = await fetch(url, {
          signal: ctrl.signal,
          headers: { Accept: 'application/octet-stream' },
        });
        if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
        if (!res.body) throw new Error('No response body for HTTP stream');

        this._connected = true;
        this._resetPlaybackClock();
        this._emit('connect', { url, transport: 'http' });

        const reader = res.body.getReader();
        // Buffer partial chunks (frames may be split across chunks)
        let carry = new Uint8Array(0);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          // Prepend any leftover bytes from previous chunk
          let chunk;
          if (carry.length > 0) {
            chunk = new Uint8Array(carry.length + value.length);
            chunk.set(carry);
            chunk.set(value, carry.length);
            carry = new Uint8Array(0);
          } else {
            chunk = value;
          }

          // Parse frames out of the chunk
          let pos = 0;
          while (pos < chunk.length) {
            if (chunk.length - pos < 12) {
              // Not enough bytes for a header yet
              carry = chunk.slice(pos);
              break;
            }
            const view       = new DataView(chunk.buffer, chunk.byteOffset + pos);
            const jointCount = view.getUint8(8);
            const facsCount  = view.getUint8(9);
            const frameSize  = 12 + jointCount * 14 + facsCount * 2;

            if (chunk.length - pos < frameSize) {
              // Incomplete frame — carry remainder to next chunk
              carry = chunk.slice(pos);
              break;
            }

            const frameBuf = chunk.buffer.slice(chunk.byteOffset + pos, chunk.byteOffset + pos + frameSize);
            this._receiveFrame(frameBuf);
            pos += frameSize;
          }
        }

        // Stream ended
        this._connected = false;
        this._emit('disconnect', { url, reason: 'stream ended' });
        if (this._autoReconnect) this._scheduleReconnect();

      } catch (err) {
        if (err.name === 'AbortError') return; // intentional disconnect
        this._connected = false;
        this._emit('error', { message: err.message });
        if (this._autoReconnect) this._scheduleReconnect();
      }
    };

    doFetch();
  }

  /** @private Parse and buffer one binary frame. */
  _receiveFrame(buffer) {
    let frame;
    try {
      frame = decodeFrame(buffer);
    } catch (e) {
      this._emit('error', { message: e.message });
      return;
    }

    // Sync playback clock on first frame
    if (this._startLocalTs === null) {
      this._startLocalTs = frame.arrivalTs;
      this._startSrvTs   = frame.serverTs;
    }

    // Update smoothed latency estimate:
    //   latency ≈ arrival - targetDelay - serverTs (mapped to local clock)
    const estimatedLatency = frame.arrivalTs - this._startLocalTs
      - (frame.serverTs - this._startSrvTs);
    this._smoothLatency = this._smoothLatency * (1 - this._smoothingAlpha)
      + Math.max(0, estimatedLatency) * this._smoothingAlpha;

    // Evict oldest frames if buffer is full
    if (this._buffer.length >= this._maxBufferFrames) {
      this._droppedFrames++;
      this._buffer.shift();
      this._emit('drop', { frameId: frame.frameId });
    }

    // Insert in order (normally frames arrive in order — fast path)
    if (this._buffer.length === 0 || frame.serverTs >= this._buffer[this._buffer.length - 1].serverTs) {
      this._buffer.push(frame);
    } else {
      // Out-of-order arrival — insert sorted by serverTs
      let i = this._buffer.length - 1;
      while (i > 0 && this._buffer[i - 1].serverTs > frame.serverTs) i--;
      this._buffer.splice(i, 0, frame);
    }

    // FPS counter
    this._totalFrames++;
    this._fpsCounter++;
    const now = performance.now();
    if (now - this._fpsTs >= 1000) {
      this._fps     = (this._fpsCounter * 1000) / (now - this._fpsTs);
      this._fpsCounter = 0;
      this._fpsTs   = now;
    }

    this._emit('data', { frameId: frame.frameId, serverTs: frame.serverTs });
  }

  /** @private */
  _interpolate(frameA, frameB, t) {
    const jA = frameA.joints, jB = frameB.joints;
    const jointCount = Math.min(jA.length, jB.length) / 7 | 0;
    const out = new Float32Array(jointCount * 7);

    for (let j = 0; j < jointCount; j++) {
      const base = j * 7;
      // Lerp position
      out[base]     = jA[base]     + (jB[base]     - jA[base])     * t;
      out[base + 1] = jA[base + 1] + (jB[base + 1] - jA[base + 1]) * t;
      out[base + 2] = jA[base + 2] + (jB[base + 2] - jA[base + 2]) * t;
      // Slerp rotation
      slerpJointQuat(jA, jB, t, out, base);
    }

    const fA = frameA.facs, fB = frameB.facs;
    const facsCount = Math.min(fA.length, fB.length);
    const facs = new Float32Array(facsCount);
    for (let f = 0; f < facsCount; f++) {
      facs[f] = fA[f] + (fB[f] - fA[f]) * t;
    }

    return {
      joints:       out,
      facs,
      frameId:      frameB.frameId,
      age:          performance.now() - frameB.arrivalTs,
      interpolated: true,
    };
  }

  /** @private */
  _makeResult(frame, interpolated) {
    return {
      joints:       frame.joints,
      facs:         frame.facs,
      frameId:      frame.frameId,
      age:          performance.now() - frame.arrivalTs,
      interpolated,
    };
  }

  /** @private */
  _resetPlaybackClock() {
    this._startLocalTs  = null;
    this._startSrvTs    = null;
    this._buffer        = [];
    this._smoothLatency = 0;
    this._fpsCounter    = 0;
    this._fpsTs         = performance.now();
  }

  /** @private */
  _scheduleReconnect() {
    this._cancelReconnect();
    this._emit('reconnecting', { url: this._url, delayMs: this._reconnectDelayMs });
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      if (this._url) this.connect(this._url, this._streamType);
    }, this._reconnectDelayMs);
  }

  /** @private */
  _cancelReconnect() {
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
  }

  /** @private */
  _emit(event, data) {
    const cbs = this._events[event];
    if (!cbs) return;
    for (const cb of cbs) {
      try { cb(data); } catch (e) { console.error('[StreamingAnimationPlayer] event handler error:', e); }
    }
  }
}
