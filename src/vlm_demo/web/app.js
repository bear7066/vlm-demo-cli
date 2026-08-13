/* The page is the clock: it plays the video and tells the server where playback is. */

const CLOCK_INTERVAL_MS = 250;

const el = {
  video: document.getElementById("video"),
  start: document.getElementById("start"),
  status: document.getElementById("status"),
  counter: document.getElementById("counter"),
  clock: document.getElementById("clock"),
  timeline: document.getElementById("timeline"),
  feed: document.getElementById("feed"),
  empty: document.getElementById("empty"),
  prompt: document.getElementById("prompt"),
  model: document.getElementById("model"),
  backend: document.getElementById("backend"),
  sampling: document.getElementById("sampling"),
};

let socket = null;
let session = null;
let started = false;
let results = 0;
const pending = new Map(); // pass index -> placeholder row awaiting its response

function send(type, extra = {}) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(
      JSON.stringify({
        type,
        t: el.video.currentTime || 0,
        playing: !el.video.paused && !el.video.ended,
        ...extra,
      }),
    );
  }
}

function setStatus(state, detail) {
  el.status.dataset.state = state;
  el.status.textContent = detail ? `${state} — ${detail}` : state;
}

function connect() {
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  socket = new WebSocket(url);
  socket.onmessage = (event) => handle(JSON.parse(event.data));
  socket.onclose = () => {
    setStatus("offline", "reconnecting");
    el.start.disabled = true;
    setTimeout(connect, 1000);
  };
}

/* ------------------------------------------------------------------ rendering */

function resetFeed() {
  el.feed.querySelectorAll(".row").forEach((row) => row.remove());
  el.timeline.replaceChildren();
  el.empty.hidden = false;
  pending.clear();
  results = 0;
}

function applySession(event) {
  session = event;
  el.prompt.textContent = event.prompt;
  el.model.textContent = event.model;
  el.backend.textContent = event.backend;
  el.sampling.textContent =
    `${event.num_frames} frames from the last ${event.window_sec}s, ` +
    `every ${event.pass_gap}s · pace=${event.pace}`;
  el.counter.textContent = `0 / ${event.total_passes} passes`;
  resetFeed();
}

function makeRow(kind, index, tStart, tEnd, text, right) {
  const row = document.createElement("div");
  row.className = `row ${kind}`;
  const meta = document.createElement("div");
  meta.className = "meta";
  const idx = document.createElement("span");
  idx.className = "idx";
  idx.textContent = `#${index}`;
  const span = document.createElement("span");
  span.textContent = `${tStart.toFixed(1)}s – ${tEnd.toFixed(1)}s`;
  meta.append(idx, span);
  if (right) {
    const extra = document.createElement("span");
    extra.textContent = right;
    meta.append(extra);
  }
  const body = document.createElement("p");
  body.className = "text";
  body.textContent = text;
  row.append(meta, body);
  return row;
}

function appendRow(row) {
  const atBottom = el.feed.scrollHeight - el.feed.scrollTop - el.feed.clientHeight < 60;
  el.empty.hidden = true;
  el.feed.append(row);
  if (atBottom) el.feed.scrollTop = el.feed.scrollHeight;
  return row;
}

function mark(tEnd, kind) {
  if (!session || !session.video.duration) return;
  const tick = document.createElement("i");
  if (kind) tick.className = kind;
  tick.style.left = `${Math.min(100, (tEnd / session.video.duration) * 100)}%`;
  el.timeline.append(tick);
}

function handle(event) {
  switch (event.type) {
    case "session":
      applySession(event);
      break;

    case "status":
      setStatus(event.state, event.detail);
      el.start.disabled = event.state === "loading" || event.state === "error";
      if (event.state === "ready" && !started) el.start.textContent = "Start";
      break;

    case "pass_started": {
      const row = makeRow(
        "pending",
        event.index,
        event.t_start,
        event.t_end,
        "thinking",
        `${event.frame_ts.length} frames`,
      );
      row.querySelector(".text").classList.add("dots");
      pending.set(event.index, appendRow(row));
      break;
    }

    case "pass_result": {
      const row = makeRow(
        event.matched ? "hit" : "",
        event.index,
        event.t_start,
        event.t_end,
        event.text,
        `${Math.round(event.latency_ms)} ms · ${event.frames_used} frames`,
      );
      const placeholder = pending.get(event.index);
      if (placeholder) {
        placeholder.replaceWith(row);
        pending.delete(event.index);
        if (el.feed.scrollHeight - el.feed.scrollTop - el.feed.clientHeight < 120) {
          el.feed.scrollTop = el.feed.scrollHeight;
        }
      } else {
        appendRow(row);
      }
      results += 1;
      if (session) el.counter.textContent = `${results} / ${session.total_passes} passes`;
      mark(event.t_end, event.matched ? "hit" : "");
      break;
    }

    case "pass_skipped": {
      const label =
        event.count > 1
          ? `skipped ${event.count} windows — ${event.reason}`
          : `skipped — ${event.reason}`;
      appendRow(makeRow("skip", event.index, event.t_start, event.t_end, label));
      mark(event.t_end, "skip");
      break;
    }

    case "error":
      appendRow(
        makeRow("err", event.index ?? 0, el.video.currentTime, el.video.currentTime, event.message),
      );
      break;
  }
}

/* ------------------------------------------------------------------ playback */

el.start.addEventListener("click", () => {
  if (!started) {
    started = true;
    resetFeed();
    el.start.textContent = "Restart";
    send("start", { t: el.video.currentTime || 0 });
    el.video.play();
  } else {
    el.video.pause();
    el.video.currentTime = 0;
    resetFeed();
    send("start", { t: 0 });
    el.video.play();
  }
});

el.video.addEventListener("play", () => started && send("resume"));
el.video.addEventListener("pause", () => started && !el.video.ended && send("pause"));
el.video.addEventListener("seeked", () => started && send("seek"));
el.video.addEventListener("ended", () => send("ended"));
el.video.addEventListener("timeupdate", () => {
  el.clock.textContent = `${el.video.currentTime.toFixed(1)}s`;
});

setInterval(() => {
  if (started) send("clock");
}, CLOCK_INTERVAL_MS);

connect();
