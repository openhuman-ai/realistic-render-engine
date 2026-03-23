#!/usr/bin/env node
/**
 * streaming-server.js — Minimal WebSocket test server for the OpenHuman streaming demo.
 *
 * Sends synthetic animation frames at ~30 FPS over WebSocket using the binary protocol
 * documented in src/animation/StreamingAnimationPlayer.js.
 *
 * Usage:
 *   npm install ws          # one-time dev dependency
 *   node demo/streaming-server.js
 *   # Then open demo/streaming.html and connect to ws://localhost:8765
 *
 * Binary frame layout (little-endian):
 *   [0..3]   uint32  serverTimestampMs
 *   [4..7]   uint32  frameId
 *   [8]      uint8   jointCount
 *   [9]      uint8   facsCount
 *   [10..11] uint16  flags (0)
 *   then jointCount × 14 bytes (3×int16 pos + 4×int16 quat)
 *   then facsCount  ×  2 bytes (int16 weight × 32767)
 */

'use strict';

import { WebSocketServer } from 'ws';
const PORT         = 8765;
const JOINT_COUNT  = 3;
const FACS_COUNT   = 52;
const FPS          = 30;
const FRAME_MS     = 1000 / FPS;

const FRAME_SIZE   = 12 + JOINT_COUNT * 14 + FACS_COUNT * 2;

let frameId  = 0;
let startMs  = Date.now();
const frameBuf = Buffer.allocUnsafe(FRAME_SIZE);
const view     = new DataView(frameBuf.buffer);

function buildFrame() {
  const elapsed = (Date.now() - startMs) / 1000;
  const serverTs = (Date.now()) >>> 0;

  view.setUint32(0, serverTs,      true);
  view.setUint32(4, frameId++ >>> 0, true);
  view.setUint8(8,  JOINT_COUNT);
  view.setUint8(9,  FACS_COUNT);
  view.setUint16(10, 0, true);

  let off = 12;
  for (let j = 0; j < JOINT_COUNT; j++) {
    // Position (mm int16)
    view.setInt16(off,     Math.round(Math.sin(elapsed * 0.7 + j * 1.2) * 50),  true); off += 2;
    view.setInt16(off,     Math.round(j * 700), true); off += 2;
    view.setInt16(off,     Math.round(Math.cos(elapsed * 0.5 + j * 0.9) * 40),  true); off += 2;
    // Rotation (normalized quaternion × 32767)
    const angle = elapsed * (0.8 + j * 0.3);
    const qy    = Math.sin(angle * 0.5), qw = Math.cos(angle * 0.5);
    view.setInt16(off,     0,                             true); off += 2;
    view.setInt16(off,     Math.round(qy * 32767),        true); off += 2;
    view.setInt16(off,     0,                             true); off += 2;
    view.setInt16(off,     Math.round(qw * 32767),        true); off += 2;
  }
  for (let f = 0; f < FACS_COUNT; f++) {
    const w = Math.max(0, Math.sin(elapsed * (0.3 + f * 0.07) + f)) * 0.5;
    view.setInt16(off, Math.round(Math.max(0, Math.min(1, w)) * 32767), true); off += 2;
  }
  return frameBuf;
}

const wss = new WebSocketServer({ port: PORT });
console.log(`[streaming-server] Listening on ws://localhost:${PORT}`);

const clients = new Set();
wss.on('connection', ws => {
  clients.add(ws);
  console.log(`[streaming-server] Client connected (total: ${clients.size})`);
  ws.on('close', () => {
    clients.delete(ws);
    console.log(`[streaming-server] Client disconnected (total: ${clients.size})`);
  });
  ws.on('error', err => console.error('[streaming-server] client error:', err.message));
});

setInterval(() => {
  if (clients.size === 0) return;
  const buf = buildFrame();
  for (const ws of clients) {
    if (ws.readyState === 1 /* OPEN */) {
      ws.send(buf);
    }
  }
}, FRAME_MS);
