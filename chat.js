/* ═══ rhombic-agent — live chat widget ═══
   Connects to local Ollama (OpenAI-compatible) running GLM-4.5
   via the Nous Hermes Agent framework.
*/

(function () {
  'use strict';

  // --- Config ---
  const API_URL = 'http://localhost:11434/v1/chat/completions';
  const MODEL = 'glm-reap';
  const SYSTEM_PROMPT = `You are rhombic-agent, an expert on lattice topology research built by Promptcrafted.

Core findings:
- The FCC lattice (12-connected, rhombic dodecahedron Voronoi cell) provides 2.3x algebraic connectivity over cubic (6-connected) with uniform weights.
- Under direction-weighted corpus edge weights, the advantage amplifies to 6.1x (Paper 2).
- FCC provides 30% shorter average paths and 40% smaller graph diameter.
- Prime-vertex mapping achieves p = 0.000025 non-accidental geometry affinity.
- 256 passing tests, Python 3.10-3.12, MPL-2.0 license.

You have 9 tools: lattice_compare, fiedler_ratio, direction_weights, spectral_analysis, prime_vertex_map, permutation_control, explain_mechanism, visualize_rd, visualize_amplification.

Papers:
- Paper 1: "The Shape of the Cell" — empirical comparison of cubic and FCC lattice topologies.
- Paper 2: "Structured Edge Weights Amplify FCC Lattice Topology Advantages via Bottleneck Resilience" — the 6.1x finding, Fiedler value amplification under heterogeneous weights.
- Paper 3: "RhombiLoRA" — geometric LoRA adapter adding 6 diagonal bridge connections to transformer attention heads. In preparation.

The rhombic dodecahedron is the Voronoi cell of the FCC lattice. It has 12 rhombic faces, 14 vertices (8 trivalent = cube corners, 6 tetravalent = octahedral bridges). Kepler proved it packs spheres optimally in 1611; Hales confirmed in 2005.

RhombiLoRA: Keep existing cubic infrastructure, add 6 diagonal bridge connections. The RD contains the cube — its 8 trivalent vertices ARE the cube's corners. The 6 tetravalent vertices are bridges converting cubic topology into space-filling FCC.

Keep answers concise, technical, and direct. Use numbers. When discussing findings, cite which paper.`;

  // --- DOM refs ---
  const chatEl = document.getElementById('agent-chat');
  const msgsEl = document.getElementById('chat-messages');
  const inputEl = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');
  const statusEl = document.getElementById('chat-status');

  if (!chatEl) return; // widget not in DOM

  let history = [{ role: 'system', content: SYSTEM_PROMPT }];
  let busy = false;

  // --- Health check ---
  async function checkHealth() {
    try {
      const res = await fetch('http://localhost:11434/api/tags', {
        method: 'GET',
        signal: AbortSignal.timeout(3000),
      });
      if (res.ok) {
        statusEl.textContent = '\u25cf online';
        statusEl.classList.add('online');
        inputEl.disabled = false;
        sendBtn.disabled = false;
        return true;
      }
    } catch (_) { /* offline */ }
    statusEl.textContent = '\u25cf offline';
    statusEl.classList.remove('online');
    return false;
  }

  // --- UI helpers ---
  function addMsg(role, text) {
    const div = document.createElement('div');
    div.className = 'msg msg-' + role;
    div.textContent = text;
    msgsEl.appendChild(div);
    msgsEl.scrollTop = msgsEl.scrollHeight;
    return div;
  }

  function addThinking() {
    const div = document.createElement('div');
    div.className = 'msg msg-thinking';
    div.textContent = 'Thinking';
    msgsEl.appendChild(div);
    msgsEl.scrollTop = msgsEl.scrollHeight;
    return div;
  }

  // --- Streaming chat ---
  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || busy) return;

    busy = true;
    sendBtn.disabled = true;
    inputEl.value = '';

    addMsg('user', text);
    history.push({ role: 'user', content: text });

    const thinkEl = addThinking();

    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: MODEL,
          messages: history,
          stream: true,
          temperature: 0.7,
          max_tokens: 1024,
        }),
      });

      if (!res.ok) {
        throw new Error('API returned ' + res.status);
      }

      // Remove thinking indicator, create response bubble
      thinkEl.remove();
      const agentDiv = addMsg('agent', '');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullText = '';
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete line

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === 'data: [DONE]') continue;
          if (!trimmed.startsWith('data: ')) continue;

          try {
            const json = JSON.parse(trimmed.slice(6));
            const delta = json.choices?.[0]?.delta?.content;
            if (delta) {
              fullText += delta;
              agentDiv.textContent = fullText;
              msgsEl.scrollTop = msgsEl.scrollHeight;
            }
          } catch (_) { /* skip malformed chunks */ }
        }
      }

      history.push({ role: 'assistant', content: fullText });

      // Keep context window manageable — trim to last 20 exchanges
      if (history.length > 42) {
        history = [history[0], ...history.slice(-40)];
      }

    } catch (err) {
      thinkEl.remove();
      addMsg('agent', 'Connection error \u2014 is Ollama running locally? (' + err.message + ')');
    }

    busy = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }

  // --- Events ---
  sendBtn.addEventListener('click', sendMessage);
  inputEl.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // --- Init ---
  checkHealth();
  setInterval(checkHealth, 15000); // poll every 15s
})();
