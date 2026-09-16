/* The page is the clock: it plays the video and tells the server where playback is.
   It also owns the library UI — which video of the --input directory is analysed, and
   uploading new ones into it or deleting ones you are done with — and which model runs.
   All of those go over REST; the resulting change comes back to every open page as a
   `session` / `library` event on the websocket. */

const CLOCK_INTERVAL_MS = 250;

const el = {
  video: document.getElementById("video"),
  novideo: document.getElementById("novideo"),
  start: document.getElementById("start"),
  status: document.getElementById("status"),
  filename: document.getElementById("filename"),
  counter: document.getElementById("counter"),
  clock: document.getElementById("clock"),
  timeline: document.getElementById("timeline"),
  feed: document.getElementById("feed"),
  empty: document.getElementById("empty"),
  prompt: document.getElementById("prompt"),
  modelInput: document.getElementById("modelInput"),
  modelList: document.getElementById("modelList"),
  modelApply: document.getElementById("modelApply"),
  modelHint: document.getElementById("modelHint"),
  sampling: document.getElementById("sampling"),
  library: document.getElementById("library"),
  libdir: document.getElementById("libdir"),
  libempty: document.getElementById("libempty"),
  videos: document.getElementById("videos"),
  uploader: document.getElementById("uploader"),
  uploadHint: document.getElementById("uploadHint"),
  pick: document.getElementById("pick"),
  file: document.getElementById("file"),
  progress: document.getElementById("progress"),
  progressBar: document.getElementById("progressBar"),
};

let socket = null;
let session = null;
let library = null;
let backendReady = false;
let started = false;
let results = 0;
let currentSrc = null;
let busy = false; // a selection, an upload or a delete is in flight
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
    return true;
  }
  return false;
}

function setStatus(state, detail) {
  el.status.dataset.state = state;
  el.status.textContent = detail ? `${state} — ${detail}` : state;
}

function connect() {
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  socket = new WebSocket(url);
  socket.onopen = refreshStartButton;
  socket.onmessage = (event) => handle(JSON.parse(event.data));
  socket.onclose = () => {
    setStatus("offline", "reconnecting");
    backendReady = false;
    refreshStartButton();
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

function refreshStartButton() {
  el.start.disabled =
    busy || !backendReady || !session || !session.video || socket?.readyState !== WebSocket.OPEN;
}

function applySession(event) {
  const video = event.video;
  session = event;
  el.prompt.textContent = event.prompt;
  applyModel(event);
  el.sampling.textContent =
    `影片每經過 ${event.pass_gap} 秒，就從前 ${event.window_sec} 秒均勻取最多 ${event.num_frames} 張畫面進行一次分析。` +
    (event.pace === "complete" ? "每個時段都會分析。" : "來不及分析的時段可能略過。");
  el.counter.textContent = `0 / ${event.total_passes} passes`;
  resetFeed();

  // A different video means a different timeline: forget where playback was.
  started = false;
  el.start.textContent = "Start";
  el.filename.textContent = video ? video.filename : "";
  el.novideo.hidden = Boolean(video);
  if (video && video.url !== currentSrc) {
    currentSrc = video.url;
    el.video.src = video.url;
    el.video.load();
  } else if (!video && currentSrc !== null) {
    currentSrc = null;
    el.video.removeAttribute("src");
    el.video.load();
  }
  refreshStartButton();
}

function formatSize(bytes) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}

function applyLibrary(event) {
  library = event;
  el.libdir.textContent = event.directory;
  el.libdir.title = event.directory;
  el.videos.replaceChildren(
    ...event.videos.map((video) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.className = "vid";
      button.dataset.name = video.name;
      button.disabled = busy;
      if (video.name === event.selected) {
        button.classList.add("active");
        button.setAttribute("aria-current", "true");
      }
      const name = document.createElement("span");
      name.className = "vname";
      name.textContent = video.name;
      name.title = video.name;
      const size = document.createElement("span");
      size.className = "vsize";
      size.textContent = formatSize(video.size_bytes);
      button.append(name, size);
      item.append(button);
      if (event.deletes_enabled) {
        const remove = document.createElement("button");
        remove.className = "vdel";
        remove.dataset.name = video.name;
        remove.disabled = busy;
        remove.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h16M10 4h4M6 7l1 13h10l1-13M10 11v6m4-6v6"/></svg>';
        remove.title = `刪除 ${video.name}`;
        remove.setAttribute("aria-label", `刪除 ${video.name}`);
        item.append(remove);
      }
      return item;
    }),
  );
  el.libempty.hidden = event.videos.length > 0;
  el.uploader.hidden = !event.uploads_enabled;
  el.uploadHint.dataset.default = `or drop files here · up to ${event.max_upload_mb} MB each`;
  if (!el.uploadHint.dataset.sticky) hint(null);
  refreshStartButton();
}

function makeRow(kind, index, tStart, tEnd, text) {
  const row = document.createElement("div");
  row.className = `row ${kind}`;
  const meta = document.createElement("div");
  meta.className = "meta";
  const idx = document.createElement("span");
  idx.className = "idx";
  idx.textContent = `#${index}`;
  const span = document.createElement("span");
  span.textContent = `影片 ${tEnd.toFixed(1)} 秒`;
  if (tStart < tEnd) span.title = `分析影格：${tStart.toFixed(1)}–${tEnd.toFixed(1)} 秒`;
  meta.append(idx, span);
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
  if (!session || !session.video || !session.video.duration) return;
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

    case "library":
      applyLibrary(event);
      break;

    case "status":
      setStatus(event.state, event.detail);
      backendReady = event.state !== "loading" && event.state !== "error";
      refreshStartButton();
      break;

    case "pass_started": {
      const row = makeRow(
        "pending",
        event.index,
        event.t_start,
        event.t_end,
        "thinking",
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

    case "error": {
      const at = event.index && session
        ? Math.min(event.index * session.pass_gap, session.video?.duration ?? Infinity)
        : el.video.currentTime;
      appendRow(
        makeRow("err", event.index ?? 0, at, at, event.message),
      );
      break;
    }
  }
}

/* ------------------------------------------------------------------ model */

/* The box is authoritative only while the user is typing in it: any session event that
   arrives (including the one our own switch triggers) writes the server's model back. */
function applyModel(event) {
  const locked = event.model_locked;
  if (document.activeElement !== el.modelInput) el.modelInput.value = event.model;
  el.modelInput.title = event.model;
  el.modelList.replaceChildren(
    ...event.available_models.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      return option;
    }),
  );
  el.modelInput.disabled = locked;
  el.modelApply.hidden = locked;
  modelHint(null);
  refreshModelControls();
}

function modelHint(message, isError = false) {
  el.modelHint.textContent = message ?? "";
  el.modelHint.classList.toggle("bad", Boolean(message) && isError);
}

function refreshModelControls() {
  const locked = Boolean(session && session.model_locked);
  el.modelInput.disabled = locked || busy;
  el.modelApply.disabled = locked || busy;
}

async function setModel(raw) {
  const name = raw.trim();
  if (busy) return;
  if (!name) {
    modelHint("enter a model id", true);
    return;
  }
  if (session && name === session.model) {
    // Already running it; tidy away whatever whitespace was typed around the name.
    el.modelInput.value = session.model;
    modelHint(null);
    return;
  }
  setBusy(true);
  el.video.pause();
  try {
    const response = await fetch("/api/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    // As with selecting a video, the server broadcasts the new session; nothing to apply here.
    // The status pill is what reports the load finishing, so a success needs no hint of its own.
    if (!response.ok) modelHint(await detail(response, `could not load ${name}`), true);
  } catch (error) {
    modelHint(String(error), true);
  } finally {
    setBusy(false);
  }
}

el.modelApply.addEventListener("click", () => setModel(el.modelInput.value));
el.modelInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") setModel(el.modelInput.value);
});

/* ------------------------------------------------------------------ library */

function hint(message, isError = false) {
  el.uploadHint.textContent = message ?? el.uploadHint.dataset.default ?? "";
  el.uploadHint.classList.toggle("bad", Boolean(message) && isError);
  el.uploadHint.dataset.sticky = message ? "1" : "";
}

function setBusy(value) {
  busy = value;
  el.videos.querySelectorAll("button").forEach((b) => (b.disabled = value));
  el.pick.disabled = value;
  refreshModelControls();
  refreshStartButton();
}

async function detail(response, fallback) {
  try {
    const body = await response.json();
    return body.detail || fallback;
  } catch {
    return fallback;
  }
}

async function selectVideo(name) {
  if (busy || (library && library.selected === name)) return;
  setBusy(true);
  el.video.pause();
  try {
    const response = await fetch("/api/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    // The server broadcasts the new session + library; nothing to apply here.
    if (!response.ok) hint(await detail(response, `could not select ${name}`), true);
    else hint(null);
  } catch (error) {
    hint(String(error), true);
  } finally {
    setBusy(false);
  }
}

async function deleteVideo(name) {
  if (busy) return;
  if (!confirm(`確定刪除 ${name}？影片也會從儲存空間刪除。`)) return;
  setBusy(true);
  try {
    const response = await fetch(`/api/videos/${encodeURIComponent(name)}`, {
      method: "DELETE",
    });
    // As with selecting, the server broadcasts the new library (and session, if the video
    // that went was the one being analysed).
    if (!response.ok) hint(await detail(response, `could not delete ${name}`), true);
    else hint(`${name} deleted`);
  } catch (error) {
    hint(String(error), true);
  } finally {
    setBusy(false);
  }
}

el.videos.addEventListener("click", (event) => {
  const remove = event.target.closest("button.vdel");
  if (remove) {
    deleteVideo(remove.dataset.name);
    return;
  }
  const button = event.target.closest("button.vid");
  if (button) selectVideo(button.dataset.name);
});

/* ------------------------------------------------------------------ uploads */

function uploadOne(file) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `/api/videos?name=${encodeURIComponent(file.name)}`);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) showProgress(event.loaded / event.total);
    };
    request.onload = () => {
      if (request.status === 201) {
        resolve(JSON.parse(request.responseText));
        return;
      }
      let message = `upload failed (${request.status})`;
      try {
        message = JSON.parse(request.responseText).detail || message;
      } catch {
        /* not JSON; keep the status line */
      }
      reject(new Error(message));
    };
    request.onerror = () => reject(new Error("upload failed: connection lost"));
    request.send(file);
  });
}

function showProgress(fraction) {
  el.progress.hidden = false;
  el.progressBar.style.width = `${Math.round(fraction * 100)}%`;
}

async function uploadFiles(files) {
  const queue = [...files];
  if (!queue.length || busy) return;
  setBusy(true);
  let stored = 0;
  try {
    for (const [n, file] of queue.entries()) {
      hint(`uploading ${file.name}${queue.length > 1 ? ` (${n + 1}/${queue.length})` : ""}…`);
      showProgress(0);
      try {
        await uploadOne(file);
        stored += 1;
      } catch (error) {
        hint(error.message, true);
        return;
      }
    }
    hint(stored === 1 ? "1 video added" : `${stored} videos added`);
  } finally {
    el.progress.hidden = true;
    setBusy(false);
  }
}

el.pick.addEventListener("click", () => el.file.click());
el.file.addEventListener("change", () => {
  uploadFiles(el.file.files);
  el.file.value = ""; // so re-picking the same file fires `change` again
});

for (const type of ["dragenter", "dragover"]) {
  el.library.addEventListener(type, (event) => {
    if (!library || !library.uploads_enabled) return;
    event.preventDefault();
    el.library.classList.add("dropping");
  });
}
for (const type of ["dragleave", "dragend"]) {
  el.library.addEventListener(type, (event) => {
    if (event.target === el.library) el.library.classList.remove("dropping");
  });
}
el.library.addEventListener("drop", (event) => {
  if (!library || !library.uploads_enabled) return;
  event.preventDefault();
  el.library.classList.remove("dropping");
  uploadFiles(event.dataTransfer.files);
});

/* ------------------------------------------------------------------ playback */

el.start.addEventListener("click", () => {
  if (!started) {
    if (!send("start", { t: el.video.currentTime || 0 })) {
      setStatus("offline", "reconnecting");
      refreshStartButton();
      return;
    }
    started = true;
    resetFeed();
    el.start.textContent = "Restart";
    el.video.play();
  } else {
    if (!send("start", { t: 0 })) {
      setStatus("offline", "reconnecting");
      refreshStartButton();
      return;
    }
    el.video.pause();
    el.video.currentTime = 0;
    resetFeed();
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
