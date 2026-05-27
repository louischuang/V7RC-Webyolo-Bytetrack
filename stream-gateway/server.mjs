import { createReadStream, existsSync, mkdirSync, readdirSync, rmSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";

const port = Number(process.env.STREAM_GATEWAY_PORT ?? 3010);
const outputDir = process.env.STREAM_OUTPUT_DIR ?? "/var/lib/v7rc-streams";
const defaultOutput = process.env.STREAM_DEFAULT_OUTPUT === "hls" ? "hls" : "mjpg";
const ytdlpFormat = process.env.YTDLP_FORMAT || "best[height<=720][ext=mp4]/best[height<=720]/best";
const ytdlpTimeoutMs = Number(process.env.YTDLP_TIMEOUT_MS ?? 45000);
const ytdlpCookiesFile = process.env.YTDLP_COOKIES_FILE || "";
const ytdlpUserAgent = process.env.YTDLP_USER_AGENT || "";
const allowedSchemes = new Set(["rtsp:", "http:", "https:"]);
const sessions = new Map();

mkdirSync(outputDir, { recursive: true });

const server = createServer(async (request, response) => {
  setCorsHeaders(response);

  if (request.method === "OPTIONS") {
    response.writeHead(204);
    response.end();
    return;
  }

  try {
    const url = new URL(request.url ?? "/", `http://${request.headers.host ?? `localhost:${port}`}`);

    if (request.method === "GET" && url.pathname === "/health") {
      sendJson(response, 200, { ok: true, sessions: sessions.size });
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/streams") {
      const body = await readJson(request);
      const created = await createStream(body, request);
      sendJson(response, 201, created);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/youtube/resolve") {
      const body = await readJson(request);
      const sourceUrl = parseHttpUrl(body?.url, "YouTube URL");
      const startedAt = performance.now();
      const resolvedUrl = await resolveYoutubeUrl(sourceUrl.href);
      const resolved = new URL(resolvedUrl);
      sendJson(response, 200, {
        ok: true,
        protocol: resolved.protocol,
        mediaHost: resolved.host,
        durationMs: Math.round(performance.now() - startedAt),
        format: ytdlpFormat,
      });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/streams") {
      sendJson(response, 200, { streams: [...sessions.values()].map(serializeSession) });
      return;
    }

    const getMatch = url.pathname.match(/^\/api\/streams\/([^/]+)$/u);
    if (request.method === "GET" && getMatch) {
      const session = sessions.get(getMatch[1]);
      if (!session) {
        sendJson(response, 404, { error: "Unknown stream id." });
        return;
      }
      sendJson(response, 200, serializeSession(session));
      return;
    }

    const deleteMatch = url.pathname.match(/^\/api\/streams\/([^/]+)$/u);
    if (request.method === "DELETE" && deleteMatch) {
      stopStream(deleteMatch[1]);
      sendJson(response, 200, { ok: true });
      return;
    }

    const mjpgMatch = url.pathname.match(/^\/streams\/([^/]+)\.mjpg$/u);
    if (request.method === "GET" && mjpgMatch) {
      streamMjpg(mjpgMatch[1], response);
      return;
    }

    const hlsMatch = url.pathname.match(/^\/streams\/([^/]+)\/(.+)$/u);
    if (request.method === "GET" && hlsMatch) {
      serveHlsFile(hlsMatch[1], hlsMatch[2], response);
      return;
    }

    sendJson(response, 404, { error: "Not found." });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown gateway error.";
    sendJson(response, 500, { error: message });
  }
});

server.listen(port, "0.0.0.0", () => {
  console.log(`stream-gateway listening on ${port}`);
});

async function createStream(body, request) {
  const sourceType = normalizeSourceType(body?.sourceType);
  const sourceUrl = sourceType === "youtube" ? parseHttpUrl(body?.url, "YouTube URL") : parseInputUrl(body?.url);
  const output = body?.output === "hls" ? "hls" : defaultOutput;
  const id = safeId(body?.id) || randomUUID();
  const resolvedUrl = sourceType === "youtube" ? await resolveYoutubeUrl(sourceUrl.href) : sourceUrl.href;
  const sessionDir = join(outputDir, id);

  stopStream(id);
  mkdirSync(sessionDir, { recursive: true });

  const session = {
    id,
    sourceType,
    inputUrl: sourceUrl.href,
    resolvedUrl,
    output,
    dir: sessionDir,
    process: null,
    clients: 0,
    status: "ready",
    logs: [],
    lastError: "",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
  sessions.set(id, session);

  if (output === "hls") {
    session.status = "starting";
    session.process = startHls(session);
  }

  return {
    id,
    sourceType,
    output,
    status: output === "hls" ? "starting" : "ready",
    url: makePublicStreamUrl(request, id, output),
  };
}

function normalizeSourceType(value) {
  if (value === "rtsp" || value === "youtube" || value === "mjpg") {
    return value;
  }

  throw new Error("sourceType must be rtsp, youtube, or mjpg.");
}

function parseInputUrl(value) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error("Stream URL is required.");
  }

  const parsed = new URL(value.trim());
  if (!allowedSchemes.has(parsed.protocol)) {
    throw new Error(`Unsupported URL scheme: ${parsed.protocol}`);
  }

  return parsed;
}

function safeId(value) {
  if (typeof value !== "string") {
    return "";
  }

  return value.trim().replace(/[^a-z0-9_-]/giu, "-").replace(/^-+|-+$/gu, "").slice(0, 80);
}

function makePublicStreamUrl(request, id, output) {
  const host = request.headers.host ?? `localhost:${port}`;
  const protocol = request.headers["x-forwarded-proto"] ?? "http";
  const base = `${protocol}://${host}`;
  return output === "hls" ? `${base}/streams/${id}/index.m3u8` : `${base}/streams/${id}.mjpg`;
}

async function resolveYoutubeUrl(url) {
  const result = await runCommand("yt-dlp", youtubeResolveArgs(url), ytdlpTimeoutMs);
  const resolved = result.stdout.split(/\r?\n/u).find(Boolean);
  if (!resolved) {
    throw new Error(result.stderr || "yt-dlp did not return a playable URL.");
  }

  return resolved;
}

function youtubeResolveArgs(url) {
  const args = ["--no-playlist", "--no-warnings", "-g", "-f", ytdlpFormat];
  if (ytdlpCookiesFile && existsSync(ytdlpCookiesFile)) {
    args.push("--cookies", ytdlpCookiesFile);
  }
  if (ytdlpUserAgent) {
    args.push("--user-agent", ytdlpUserAgent);
  }
  args.push(url);
  return args;
}

function parseHttpUrl(value, label) {
  const parsed = parseInputUrl(value);
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(`${label} must use http:// or https://.`);
  }

  return parsed;
}

function startHls(session) {
  const args = [
    ...inputArgs(session),
    "-an",
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-tune",
    "zerolatency",
    "-f",
    "hls",
    "-hls_time",
    "1",
    "-hls_list_size",
    "5",
    "-hls_flags",
    "delete_segments+append_list+omit_endlist",
    join(session.dir, "index.m3u8"),
  ];
  const child = spawnFfmpeg(args);
  child.stderr.on("data", (chunk) => {
    appendLog(session, chunk.toString());
  });
  session.status = "running";
  session.updatedAt = new Date().toISOString();
  child.on("exit", () => {
    const current = sessions.get(session.id);
    if (current?.process === child) {
      current.process = null;
      current.status = "stopped";
      current.updatedAt = new Date().toISOString();
    }
  });
  return child;
}

function streamMjpg(id, response) {
  const session = sessions.get(id);
  if (!session) {
    sendJson(response, 404, { error: "Unknown stream id." });
    return;
  }

  const args = [
    ...inputArgs(session),
    "-an",
    "-vf",
    "fps=10",
    "-q:v",
    "5",
    "-f",
    "mpjpeg",
    "pipe:1",
  ];
  const child = spawnFfmpeg(args);
  session.clients += 1;
  session.status = "running";
  session.updatedAt = new Date().toISOString();

  response.writeHead(200, {
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": "no-store",
    "Connection": "close",
    "Content-Type": "multipart/x-mixed-replace; boundary=ffmpeg",
  });

  child.stdout.pipe(response);
  child.stderr.on("data", (chunk) => {
    appendLog(session, chunk.toString());
  });
  child.on("exit", () => {
    response.end();
  });
  response.on("close", () => {
    child.kill("SIGTERM");
    const current = sessions.get(id);
    if (current) {
      current.clients = Math.max(0, current.clients - 1);
      current.status = current.clients > 0 ? "running" : "ready";
      current.updatedAt = new Date().toISOString();
    }
  });
}

function serveHlsFile(id, fileName, response) {
  const session = sessions.get(id);
  if (!session) {
    sendJson(response, 404, { error: "Unknown stream id." });
    return;
  }

  const cleanName = normalize(fileName).replace(/^(\.\.(\/|\\|$))+/u, "");
  const filePath = join(session.dir, cleanName);
  if (!filePath.startsWith(session.dir) || !existsSync(filePath) || !statSync(filePath).isFile()) {
    sendJson(response, 404, { error: "Stream file not ready." });
    return;
  }

  response.writeHead(200, {
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": "no-store",
    "Content-Type": contentType(filePath),
  });
  createReadStream(filePath).pipe(response);
}

function stopStream(id) {
  const session = sessions.get(id);
  if (!session) {
    return;
  }

  session.process?.kill("SIGTERM");
  sessions.delete(id);
  rmSync(session.dir, { recursive: true, force: true });
}

function serializeSession(session) {
  return {
    id: session.id,
    sourceType: session.sourceType,
    output: session.output,
    status: session.status,
    clients: session.clients,
    createdAt: session.createdAt,
    updatedAt: session.updatedAt,
    inputUrl: maskUrl(session.inputUrl),
    logs: session.logs.slice(-10),
    lastError: session.lastError,
  };
}

function appendLog(session, text) {
  const lines = text
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) {
    return;
  }

  for (const line of lines) {
    console.warn(`[${session.id}] ffmpeg: ${line}`);
    session.logs.push({
      at: new Date().toISOString(),
      message: line,
    });
    session.lastError = line;
  }
  session.logs = session.logs.slice(-50);
  session.updatedAt = new Date().toISOString();
}

function maskUrl(value) {
  try {
    const parsed = new URL(value);
    if (parsed.username || parsed.password) {
      parsed.username = parsed.username ? "***" : "";
      parsed.password = parsed.password ? "***" : "";
    }
    return parsed.href;
  } catch {
    return "";
  }
}

function inputArgs(session) {
  const url = session.resolvedUrl;
  const parsed = new URL(url);
  if (parsed.protocol === "rtsp:") {
    return ["-rtsp_transport", "tcp", "-i", url];
  }

  if (session.sourceType === "youtube") {
    return ["-re", "-i", url];
  }

  return ["-i", url];
}

function spawnFfmpeg(args) {
  return spawn("ffmpeg", ["-hide_banner", "-loglevel", "warning", ...args], {
    stdio: ["ignore", "pipe", "pipe"],
  });
}

function runCommand(command, args, timeoutMs) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`${command} timed out.`));
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      clearTimeout(timer);
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }

      reject(new Error(stderr || `${command} exited with code ${code}.`));
    });
  });
}

function readJson(request) {
  return new Promise((resolve, reject) => {
    let data = "";
    request.on("data", (chunk) => {
      data += chunk;
      if (data.length > 1024 * 1024) {
        reject(new Error("Request body is too large."));
        request.destroy();
      }
    });
    request.on("end", () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch {
        reject(new Error("Request body must be JSON."));
      }
    });
    request.on("error", reject);
  });
}

function sendJson(response, status, payload) {
  response.writeHead(status, {
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": "no-store",
    "Content-Type": "application/json; charset=utf-8",
  });
  response.end(JSON.stringify(payload));
}

function setCorsHeaders(response) {
  response.setHeader("Access-Control-Allow-Origin", "*");
  response.setHeader("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS");
  response.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

function contentType(filePath) {
  const ext = extname(filePath);
  if (ext === ".m3u8") {
    return "application/vnd.apple.mpegurl";
  }
  if (ext === ".ts") {
    return "video/mp2t";
  }

  return "application/octet-stream";
}

process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);

function shutdown() {
  for (const id of [...sessions.keys()]) {
    stopStream(id);
  }
  for (const entry of readdirSync(outputDir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      rmSync(join(outputDir, entry.name), { recursive: true, force: true });
    }
  }
  process.exit(0);
}
