"""Check the rendered Voice tab and execute its STT handler without network."""

import shutil
import subprocess
from html.parser import HTMLParser

import pytest

from api.admin import _render_partial


class _VoiceMarkup(HTMLParser):
    data_expression = ""
    save_expression = ""

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "div" and "x-data" in attributes:
            self.data_expression = attributes["x-data"] or ""
        if tag == "button":
            handler = attributes.get("@click") or ""
            if "'voice.tts_enabled'" in handler:
                self.save_expression = handler


@pytest.fixture
def voice_markup():
    response = _render_partial("partials/identity.html", stt_api_base_url="http://whistle:8000/v1")
    markup = _VoiceMarkup()
    markup.feed(response.body.decode())
    markup.close()
    return markup


def test_voice_handlers_survive_html_parsing(voice_markup):
    assert voice_markup.data_expression.rstrip().endswith("}")
    assert voice_markup.save_expression.rstrip().endswith("})")


def test_stt_handler_results_and_timeout(voice_markup):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js required to execute the frontend handler")
    script = r"""
const assert = require('node:assert/strict');
let fireTimeout, cleared, pendingSignal;
global.localStorage = { getItem: () => '' };
global.setTimeout = (fn, delay) => {
  assert.equal(delay, 320000); fireTimeout = fn; return 42;
};
global.clearTimeout = id => { assert.equal(id, 42); cleared = true; };
const state = DATA_EXPRESSION;
(async () => {
  for (const [ok, body, expected] of [
    [true, { ok: true, mode: 'remote', text: 'zero one two', duration_ms: 10 }, '"zero one two"'],
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
""".replace("DATA_EXPRESSION", voice_markup.data_expression)
    subprocess.run([node, "-e", script], check=True, timeout=10)
