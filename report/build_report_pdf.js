#!/usr/bin/env node
/**
 * Replace [END] with the current UTC time (HH:MM),
 * then print the report HTML to PDF with headless Chrome (DevTools
 * Page.printToPDF, CSS page size Letter, page numbers in the footer).
 * Usage: node build_report_pdf.js <report.html> <report.pdf> <profile-dir>
 */
const fs = require("fs");
const { spawn } = require("child_process");
const [HTML, PDF, PROFILE] = process.argv.slice(2);
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9335;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  const now = new Date();
  const hhmm = now.toISOString().slice(11, 16);
  const src = fs.readFileSync(HTML, "utf8");
  if (src.includes("[END]")) fs.writeFileSync(HTML, src.replace("[END]", hhmm));
  console.log(`build time (UTC): ${now.toISOString()}; [END] -> ${src.includes("[END]") ? hhmm : "(already replaced)"}`);
  const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`, "--no-first-run", "--no-default-browser-check", "--disable-gpu", "about:blank"], { stdio: "ignore" });
  let targets;
  for (let i = 0; i < 50; i += 1) { try { targets = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json(); break; } catch { await sleep(200); } }
  const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r));
  let id = 0; const pending = new Map();
  ws.addEventListener("message", (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  await send("Page.enable");
  await send("Page.navigate", { url: "file://" + HTML });
  for (let i = 0; i < 100; i += 1) {
    const ready = await send("Runtime.evaluate", { expression: "document.readyState === 'complete' && document.fonts.status === 'loaded'", returnByValue: true });
    if (ready.result.result.value) break;
    await sleep(100);
  }
  const res = await send("Page.printToPDF", {
    preferCSSPageSize: true, printBackground: true, displayHeaderFooter: true,
    headerTemplate: "<div></div>",
    footerTemplate: '<div style="width:100%;text-align:center;font-family:Times New Roman,serif;font-size:8pt;"><span class="pageNumber"></span></div>',
  });
  if (!res.result) throw new Error(JSON.stringify(res.error));
  fs.writeFileSync(PDF, Buffer.from(res.result.data, "base64"));
  console.log(`wrote ${PDF} (${fs.statSync(PDF).size} bytes)`);
  await send("Browser.close").catch(() => {});
  setTimeout(() => chrome.kill("SIGKILL"), 500);
})().catch((e) => { console.error("BUILD FAILED", e); process.exit(1); });
