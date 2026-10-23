"""Execute the Voice tab handler with browser APIs stubbed (no network)."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_stt_handler_results_and_timeout():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required to execute the frontend handler")
    template = (Path(__file__).parents[1] / "api/templates/partials/identity.html").read_text()
    handler = template.split("  testStt() {", 1)[1].split("\n  }\n}", 1)[0]
    script = r"""
const assert = require('node:assert/strict');
let fireTimeout, cleared, pendingSignal;
global.localStorage = { getItem: () => '' };
global.setTimeout = (fn, delay) => {
  assert.equal(delay, 320000); fireTimeout = fn; return 42;
};
global.clearTimeout = id => { assert.equal(id, 42); cleared = true; };
const state = { sttMode: 'remote', testStt: function() { HANDLER } };
(async () => {
  for (const [ok, body, expected] of [
    [true, { ok: true, mode: 'remote', text: 'zero one two', duration_ms: 10 }, 'zero one two'],
    [true, { ok: false, mode: 'remote', error: 'STT failed: unavailable' },
      'STT failed: unavailable'],
    [false, { detail: 'Test audio failed: missing' }, 'Test audio failed: missing'],
  ]) {
    cleared = false;
    global.fetch = async () => ({ ok, json: async () => body });
    await state.testStt();
    assert(state.sttTestResult.includes(expected));
    assert(cleared);
    assert.equal(state.sttTestOk, !!body.ok && ok);
  }
  global.fetch = (_, options) => new Promise((resolve, reject) => {
    pendingSignal = options.signal;
    pendingSignal.addEventListener('abort', () =>
      reject(new DOMException('aborted', 'AbortError')));
  });
  cleared = false;
  const request = state.testStt();
  fireTimeout();
  await request;
  assert(pendingSignal.aborted);
  assert.equal(state.sttTestOk, false);
  assert(state.sttTestResult.includes('timed out after 320 seconds'));
  assert(cleared);
})().catch(err => { console.error(err); process.exitCode = 1; });
""".replace("HANDLER", handler)
    subprocess.run([node, "-e", script], check=True, timeout=10)
