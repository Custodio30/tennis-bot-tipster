# app.py — Professional Dashboard (Flask + Tailwind + SSE)
# Run:
#   pip install flask pyyaml
#   python app.py
# Open: http://127.0.0.1:8000

from __future__ import annotations
import os, sys, csv, json, pathlib, subprocess
from typing import Dict, Any, List, Iterable
from flask import Flask, request, jsonify, send_from_directory, Response

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable  # current venv python
SETTINGS_JSON = HERE / "configs" / "ui_settings.json"

app = Flask(__name__)

# ----------------------- helpers -----------------------

def ensure_parent(path: str | pathlib.Path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

def _proc_env() -> dict:
    """Ambiente para subprocessos com UTF-8 forçado (evita UnicodeEncodeError no Windows)."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env

def run_cmd(args: List[str], cwd: pathlib.Path | None = None) -> Dict[str, Any]:
    if cwd is None:
        cwd = HERE
    try:
        p = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            shell=False,
            env=_proc_env(),
            encoding="utf-8",
            errors="replace",
        )
        return {"cmd": args, "code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as e:
        return {"cmd": args, "code": -1, "stdout": "", "stderr": str(e)}

def stream_cmd(args: List[str], cwd: pathlib.Path | None = None) -> Iterable[str]:
    """SSE stream of a process' stdout/stderr in real time."""
    if cwd is None:
        cwd = HERE
    yield f"data: {json.dumps({'event':'start','cmd':args})}\n\n"
    try:
        p = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_proc_env(),
            encoding="utf-8",
            errors="replace",
        )
        assert p.stdout is not None
        for line in p.stdout:
            yield f"data: {json.dumps({'event':'log','line':line.rstrip()})}\n\n"
        p.wait()
        yield f"data: {json.dumps({'event':'end','code':p.returncode})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'event':'error','message':str(e)})}\n\n"

# ----------------------- settings (persist on disk) -----------------------

def read_settings() -> Dict[str, Any]:
    if SETTINGS_JSON.exists():
        try:
            return json.loads(SETTINGS_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def write_settings(obj: Dict[str, Any]):
    ensure_parent(SETTINGS_JSON)
    SETTINGS_JSON.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

# ----------------------- API: info & files -----------------------

@app.get("/api/health")
def api_health():
    return jsonify({"status":"ok","cwd":str(HERE),"python":sys.version.split(" ")[0]})

@app.get("/api/list")
def api_list():
    def ls(rel: str):
        base = HERE / rel
        items: List[Dict[str, Any]] = []
        if base.exists():
            for p in base.rglob("*.csv"):
                items.append({"name": str(p.relative_to(HERE)), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)})
        return sorted(items, key=lambda x: x["mtime"], reverse=True)
    return jsonify({
        "fixtures": ls("data/fixtures"),
        "outputs": ls("outputs"),
        "processed": ls("data/processed"),
        "news": ls("data/news"),
    })

@app.get("/api/preview_csv")
def api_preview_csv():
    path = request.args.get("path")
    n = int(request.args.get("n", 50))
    if not path: return jsonify({"error":"path is required"}), 400
    abs_path = (HERE / path).resolve()
    if not abs_path.exists(): return jsonify({"error": f"file not found: {path}"}), 404
    rows, cols = [], []
    with abs_path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        try: cols = next(rdr)
        except StopIteration: cols = []
        for i, row in enumerate(rdr):
            if i >= n: break
            rows.append(row)
    return jsonify({"columns": cols, "rows": rows, "path": str(abs_path.relative_to(HERE))})

@app.get("/download/<path:relpath>")
def download(relpath: str):
    full = (HERE / relpath).resolve()
    if not full.exists() or not full.is_file():
        return Response("Not found", status=404)
    return send_from_directory(directory=str(full.parent), path=full.name, as_attachment=True)

# ----------------------- API: pipeline (sync) -----------------------

@app.post("/api/fetch")
def api_fetch():
    d = request.get_json(force=True)
    provider = d.get("provider", "sofascore_playwright")
    days = str(d.get("days", 2))
    out = d.get("out", r"data\fixtures\latest.csv")
    ensure_parent(out)
    args = [PY, "scripts/fetch_fixtures_sofascore.py", "--provider", provider, "--days", days, "--out", out]
    return jsonify(run_cmd(args))

@app.post("/api/prep")
def api_prep():
    d = request.get_json(force=True)
    src = d.get("src", r"data\fixtures\latest.csv")
    out = d.get("out", r"data\fixtures\latest_for_tips.csv")
    ensure_parent(out)
    args = [PY, "scripts/prep_fixtures_for_tips.py", src, out]
    return jsonify(run_cmd(args))

@app.post("/api/tips")
def api_tips():
    d = request.get_json(force=True)
    history = d.get("history", r"data/processed/matches.csv")
    fixtures = d.get("fixtures", r"data/fixtures/latest_for_tips.csv")
    config = d.get("config", r"configs/default.yaml")
    model_path = d.get("model_path", r"models/model.joblib")
    out = d.get("out", r"outputs/tips.csv")
    ensure_parent(out)
    args = [PY, "-m", "src.tennistips.cli", "tips", "--history", history, "--fixtures", fixtures, "--config", config, "--model-path", model_path, "--out", out]
    return jsonify(run_cmd(args))

@app.post("/api/filter")
def api_filter():
    d = request.get_json(force=True)
    src = d.get("src", r"outputs/tips.csv")
    out = d.get("out", r"outputs/tips_filtered.csv")
    news = d.get("news")  # ex.: "data/news/news_flags.csv"
    min_prob = str(d.get("min_prob", 0.60))
    penalty = str(d.get("penalty", 0.35))
    half_life = str(d.get("half_life", 7))
    ensure_parent(out)
    args = [PY, "scripts/filter_tips.py", src, out, "--min-prob", min_prob, "--penalty", penalty, "--half-life", half_life]
    if news:
        args.extend(["--news", news])
    return jsonify(run_cmd(args))

# ----------------------- API: pipeline (SSE streaming) -----------------------

@app.get("/api/stream/fetch")
def sse_fetch():
    provider = request.args.get("provider", "sofascore_playwright")
    days = request.args.get("days", "2")
    out = request.args.get("out", r"data\fixtures\latest.csv")
    ensure_parent(out)
    args = [PY, "scripts/fetch_fixtures_sofascore.py", "--provider", provider, "--days", days, "--out", out]
    return Response(stream_cmd(args), mimetype='text/event-stream')

@app.get("/api/stream/prep")
def sse_prep():
    src = request.args.get("src", r"data\fixtures\latest.csv")
    out = request.args.get("out", r"data\fixtures\latest_for_tips.csv")
    ensure_parent(out)
    args = [PY, "scripts/prep_fixtures_for_tips.py", src, out]
    return Response(stream_cmd(args), mimetype='text/event-stream')

@app.get("/api/stream/tips")
def sse_tips():
    history = request.args.get("history", r"data/processed/matches.csv")
    fixtures = request.args.get("fixtures", r"data/fixtures/latest_for_tips.csv")
    config = request.args.get("config", r"configs/default.yaml")
    model_path = request.args.get("model_path", r"models/model.joblib")
    out = request.args.get("out", r"outputs/tips.csv")
    ensure_parent(out)
    args = [PY, "-m", "src.tennistips.cli", "tips", "--history", history, "--fixtures", fixtures, "--config", config, "--model-path", model_path, "--out", out]
    return Response(stream_cmd(args), mimetype='text/event-stream')

@app.get("/api/stream/filter")
def sse_filter():
    src = request.args.get("src", r"outputs/tips.csv")
    out = request.args.get("out", r"outputs/tips_filtered.csv")
    news = request.args.get("news", None)
    min_prob = request.args.get("min_prob", "0.60")
    penalty = request.args.get("penalty", "0.35")
    half_life = request.args.get("half_life", "7")
    ensure_parent(out)
    args = [PY, "scripts/filter_tips.py", src, out, "--min-prob", min_prob, "--penalty", penalty, "--half-life", half_life]
    if news:
        args.extend(["--news", news])
    return Response(stream_cmd(args), mimetype='text/event-stream')

# ----------------------- API: settings & uploads -----------------------

@app.get("/api/get_settings")
def api_get_settings():
    return jsonify(read_settings())

@app.post("/api/save_settings")
def api_save_settings():
    data = request.get_json(force=True)
    write_settings(data or {})
    return jsonify({"ok": True})

@app.post("/api/upload")
def api_upload():
    f = request.files.get('file')
    dest = request.form.get('dest', '')
    if not f or not dest:
        return jsonify({"error":"file and dest are required"}), 400
    abs_dest = (HERE / dest).resolve()
    ensure_parent(abs_dest)
    f.save(str(abs_dest))
    return jsonify({"ok": True, "saved": str(abs_dest.relative_to(HERE))})

# ----------------------- UI -----------------------

BASE_CSS = """
  <script src="https://cdn.tailwindcss.com"></script>
  <script>tailwind.config={theme:{extend:{colors:{brand:'#0d6efd'}}}}</script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root{color-scheme:light dark}
    body{font-family:Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif}
    .card{ @apply bg-white/90 dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-2xl p-5 shadow-sm; }
    .btn{ @apply inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold; }
    .btn-primary{ @apply bg-brand text-white hover:bg-blue-600; }
    .btn-ghost{ @apply bg-white dark:bg-neutral-800 text-brand border border-brand/30; }
    .badge{ @apply inline-block text-xs px-2 py-0.5 rounded-full border; }
    .link{ @apply text-brand underline underline-offset-4; }
    .sidebar a{ @apply block px-3 py-2 rounded-lg text-sm font-medium hover:bg-brand/10; }
    .sidebar a.active{ @apply bg-brand/10 text-brand; }
    .kbd{ @apply px-2 py-1 rounded border text-xs bg-neutral-100 dark:bg-neutral-800; }
  </style>
"""

NAVBAR = """
  <header class=\"sticky top-0 z-40 backdrop-blur bg-white/70 dark:bg-neutral-950/70 border-b border-neutral-200 dark:border-neutral-800\">
    <div class=\"max-w-7xl mx-auto px-4 py-3 flex items-center justify-between\">
      <div class=\"flex items-center gap-3\">
        <div class=\"w-8 h-8 rounded-xl bg-brand flex items-center justify-center text-white font-bold\">🎾</div>
        <a href=\"/\" class=\"text-lg font-bold\">Tennis Tipster</a>
        <span id=\"health\" class=\"badge ml-3\">checking…</span>
      </div>
      <nav class=\"hidden md:flex items-center gap-4 text-sm\">
        <a class=\"hover:underline\" href=\"/pipeline\">Pipeline</a>
        <a class=\"hover:underline\" href=\"/files\">Ficheiros</a>
        <a class=\"hover:underline\" href=\"/settings\">Definições</a>
        <a class=\"hover:underline\" href=\"/about\">Sobre</a>
      </nav>
      <div class=\"flex items-center gap-3 text-sm\">
        <button id=\"themeBtn\" class=\"btn btn-ghost\">🌓 Tema</button>
        <a class=\"btn btn-ghost\" href=\"/download/outputs/tips_filtered.csv\">⬇️ Tips</a>
      </div>
    </div>
  </header>
"""

SIDEBAR = """
  <aside class=\"sidebar hidden lg:block w-64 p-4\">
    <div class=\"card\">
      <nav class=\"space-y-1\">
        <a href=\"/pipeline\" data-match=\"/pipeline\">⚙️ Pipeline</a>
        <a href=\"/files\" data-match=\"/files\">📁 Ficheiros</a>
        <a href=\"/settings\" data-match=\"/settings\">⚙️ Definições</a>
        <a href=\"/about\" data-match=\"/about\">ℹ️ Sobre</a>
      </nav>
    </div>
  </aside>
"""

BASE_JS = """
  <script>
    const $ = (s)=>document.querySelector(s);
    const setBadge=(ok)=>{const el=$('#health'); if(!el) return; el.textContent= ok?'online':'offline'; el.className='badge '+(ok?'border-green-500 text-green-600':'border-red-500 text-red-600');};
    async function health(){ try{ const r=await fetch('/api/health'); setBadge(r.ok);}catch(e){ setBadge(false);} }
    function toggleTheme(){ const d=document.documentElement; const dark=d.classList.toggle('dark'); localStorage.setItem('tt_theme', dark?'dark':'light'); }
    function initTheme(){ const t=localStorage.getItem('tt_theme'); if(t==='dark') document.documentElement.classList.add('dark'); }
    function activateSidebar(){ const path=location.pathname; document.querySelectorAll('.sidebar a').forEach(a=>{ if(path.startsWith(a.dataset.match)) a.classList.add('active'); }); }
    window.addEventListener('DOMContentLoaded', ()=>{ initTheme(); health(); activateSidebar(); var tb=document.getElementById('themeBtn'); if(tb){ tb.addEventListener('click', toggleTheme); } });
  </script>
"""

# ----------------------- pages -----------------------

def layout(content_html: str, title: str = "Dashboard") -> str:
    return f"""
<!doctype html>
<html lang=pt>
<head>
  <meta charset=utf-8>
  <meta name=viewport content="width=device-width, initial-scale=1" />
  <title>Tennis Tipster • {title}</title>
  {BASE_CSS}
</head>
<body class="bg-neutral-100 dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100">
  {NAVBAR}
  <main class="max-w-7xl mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-[16rem_1fr] gap-6">
    {SIDEBAR}
    <section>
      {content_html}
    </section>
  </main>
  {BASE_JS}
</body>
</html>
"""

# ---------- index ----------
@app.get("/")
def home():
    html = """
    <div class=card>
      <h2 class="text-xl font-semibold mb-2">Bem-vindo 👋</h2>
      <p class=text-sm>Use a barra lateral para navegar. Comece em <a class=link href=/pipeline>Pipeline</a>.</p>
    </div>
    <div class="grid md:grid-cols-2 gap-6 mt-6">
      <div class=card>
        <h3 class="font-semibold mb-2">Atalhos rápidos</h3>
        <div class="flex gap-2 flex-wrap">
          <a class="btn btn-primary" href="/pipeline">Executar Pipeline</a>
          <a class="btn btn-ghost" href="/files">Ver Ficheiros</a>
          <a class="btn btn-ghost" href="/settings">Definições</a>
        </div>
      </div>
      <div class=card>
        <h3 class="font-semibold mb-2">Estado</h3>
        <div id=status class=text-sm>—</div>
      </div>
    </div>
    <script>
      (async function(){
        try{
          const r=await fetch('/api/health'); const j=await r.json();
          document.getElementById('status').textContent = 'Python '+(j.python||'?')+' • cwd '+(j.cwd||'?');
        }catch(e){ document.getElementById('status').textContent='offline'; }
      })();
    </script>
    """
    return layout(html, title="Início")

# ---------- pipeline ----------
@app.get("/pipeline")
def page_pipeline():
    settings = read_settings() or {}
    def g(k, v): return str(settings.get(k, v))

    html_tpl = r"""
    <div class=card>
      <div class="flex items-center justify-between mb-3">
        <h2 class="font-semibold text-lg">Pipeline</h2>
        <button id=btnAll class="btn btn-ghost">▶️ Executar Tudo</button>
      </div>
      <div class="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class=card>
          <h3 class="font-semibold mb-2">1) Buscar Fixtures</h3>
          <label class=text-xs>Provider</label>
          <input id=fetch_provider class="w-full border rounded-lg px-3 py-2 mb-2" value="%%FETCH_PROVIDER%%" />
          <label class=text-xs>Dias</label>
          <input id=fetch_days type=number class="w-full border rounded-lg px-3 py-2 mb-2" value="%%FETCH_DAYS%%" />
          <label class=text-xs>Out CSV</label>
          <input id=fetch_out class="w-full border rounded-lg px-3 py-2 mb-3" value="%%FETCH_OUT%%" />
          <div class="flex gap-2">
            <button id=btnFetch class="btn btn-primary">Executar Fetch</button>
            <button id=btnFetchSSE class="btn btn-ghost">SSE</button>
          </div>
          <pre id=log_fetch class="mt-2 text-xs text-neutral-500 whitespace-pre-wrap h-28 overflow-auto"></pre>
        </div>

        <div class=card>
          <h3 class="font-semibold mb-2">2) Preparar Fixtures</h3>
          <label class=text-xs>Input CSV</label>
          <input id=prep_src class="w-full border rounded-lg px-3 py-2 mb-2" value="%%PREP_SRC%%" />
          <label class=text-xs>Out CSV</label>
          <input id=prep_out class="w-full border rounded-lg px-3 py-2 mb-3" value="%%PREP_OUT%%" />
          <div class="flex gap-2">
            <button id=btnPrep class="btn btn-primary">Executar Prep</button>
            <button id=btnPrepSSE class="btn btn-ghost">SSE</button>
          </div>
          <pre id=log_prep class="mt-2 text-xs text-neutral-500 whitespace-pre-wrap h-28 overflow-auto"></pre>
        </div>

        <div class=card>
          <h3 class="font-semibold mb-2">3) Gerar Tips</h3>
          <label class=text-xs>History</label>
          <input id=tips_hist class="w-full border rounded-lg px-3 py-2 mb-2" value="%%TIPS_HIST%%" />
          <label class=text-xs>Fixtures</label>
          <input id=tips_fx class="w-full border rounded-lg px-3 py-2 mb-2" value="%%TIPS_FX%%" />
          <label class=text-xs>Config</label>
          <input id=tips_cfg class="w-full border rounded-lg px-3 py-2 mb-2" value="%%TIPS_CFG%%" />
          <label class=text-xs>Model</label>
          <input id=tips_model class="w-full border rounded-lg px-3 py-2 mb-3" value="%%TIPS_MODEL%%" />
          <label class=text-xs>Out CSV</label>
          <input id=tips_out class="w-full border rounded-lg px-3 py-2 mb-3" value="%%TIPS_OUT%%" />
          <div class="flex gap-2">
            <button id=btnTips class="btn btn-primary">Executar Tips</button>
            <button id=btnTipsSSE class="btn btn-ghost">SSE</button>
          </div>
          <pre id=log_tips class="mt-2 text-xs text-neutral-500 whitespace-pre-wrap h-28 overflow-auto"></pre>
        </div>

        <div class=card>
          <h3 class="font-semibold mb-2">4) Filtrar Tips</h3>
          <label class=text-xs>Input CSV</label>
          <input id=flt_src class="w-full border rounded-lg px-3 py-2 mb-2" value="%%FLT_SRC%%" />
          <label class=text-xs>Out CSV</label>
          <input id=flt_out class="w-full border rounded-lg px-3 py-2 mb-2" value="%%FLT_OUT%%" />

          <label class=text-xs>News CSV</label>
          <input id=flt_news class="w-full border rounded-lg px-3 py-2 mb-2" value="%%FLT_NEWS%%" />

          <div class="grid grid-cols-3 gap-2 mb-2">
            <div>
              <label class=text-xs>Min Prob</label>
              <input id=flt_minprob type=number step="0.01" class="w-full border rounded-lg px-3 py-2" value="%%FLT_MINPROB%%" />
            </div>
            <div>
              <label class=text-xs>Penalty</label>
              <input id=flt_penalty type=number step="0.01" class="w-full border rounded-lg px-3 py-2" value="%%FLT_PENALTY%%" />
            </div>
            <div>
              <label class=text-xs>Half-life (dias)</label>
              <input id=flt_halflife type=number class="w-full border rounded-lg px-3 py-2" value="%%FLT_HALFLIFE%%" />
            </div>
          </div>

          <div class="flex gap-2">
            <button id=btnFilter class="btn btn-primary">Executar Filtro</button>
            <button id=btnFilterSSE class="btn btn-ghost">SSE</button>
          </div>
          <pre id=log_filter class="mt-2 text-xs text-neutral-500 whitespace-pre-wrap h-28 overflow-auto"></pre>
        </div>
      </div>
    </div>

    <div class=card>
      <div class="flex items-center justify-between mb-3">
        <h3 class="font-semibold">Logs Consolidados</h3>
        <div class="flex items-center gap-2 text-xs">
          <button id=btnSave class="btn btn-ghost">Guardar definições</button>
          <button id=btnClear class="btn btn-ghost">Limpar logs</button>
        </div>
      </div>
      <pre id=logs class="bg-black text-green-400 p-4 rounded-xl h-72 overflow-auto">(sem logs)</pre>
    </div>

    <script>
      const $=(s)=>document.querySelector(s); const L=document.getElementById('logs');
      function log(id,msg){
        const pre=document.getElementById(id);
        if(pre){ pre.textContent += (msg+'\\n'); pre.scrollTop = pre.scrollHeight; }
        L.textContent = '[' + new Date().toLocaleTimeString() + '] ' + id + ': ' + msg + '\\n' + L.textContent;
      }
      async function postJSON(url,p){ const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)}); return r.json(); }
      function sse(url, id){
        const es=new EventSource(url); log(id, '--- streaming ---');
        es.onmessage=(ev)=>{
          try{
            const j=JSON.parse(ev.data);
            if(j.event==='log') log(id,j.line);
            if(j.event==='end'){ log(id,'[exit '+j.code+']'); es.close(); }
            if(j.event==='error'){ log(id,'[error] '+j.message); es.close(); }
          }catch(e){ log(id, ev.data); }
        };
        es.onerror=()=>{ log(id,'[sse error]'); es.close(); };
      }
      function enc(v){ return encodeURIComponent(v); }
      function buildQS(o){ return Object.entries(o).map(function(kv){return kv[0] + '=' + enc(kv[1]);}).join('&'); }

      async function runFetch(){ const payload={}; payload.provider=$('#fetch_provider').value; payload.days=Number($('#fetch_days').value||2); payload.out=$('#fetch_out').value; log('log_fetch', JSON.stringify(payload)); const j=await postJSON('/api/fetch', payload); log('log_fetch', (j.code===0?'OK':'ERR')+"\\n"+(j.stdout||'')+"\\n"+(j.stderr||'')); }
      async function runPrep(){ const payload={}; payload.src=$('#prep_src').value; payload.out=$('#prep_out').value; log('log_prep', JSON.stringify(payload)); const j=await postJSON('/api/prep', payload); log('log_prep', (j.code===0?'OK':'ERR')+"\\n"+(j.stdout||'')+"\\n"+(j.stderr||'')); }
      async function runTips(){ const payload={}; payload.history=$('#tips_hist').value; payload.fixtures=$('#tips_fx').value; payload.config=$('#tips_cfg').value; payload.model_path=$('#tips_model').value; payload.out=$('#tips_out').value; log('log_tips', JSON.stringify(payload)); const j=await postJSON('/api/tips', payload); log('log_tips', (j.code===0?'OK':'ERR')+"\\n"+(j.stdout||'')+"\\n"+(j.stderr||'')); }

      async function runFilter(){
        const payload={};
        payload.src=$('#flt_src').value;
        payload.out=$('#flt_out').value;
        payload.news=$('#flt_news').value || null;
        payload.min_prob=Number($('#flt_minprob').value || 0.60);
        payload.penalty=Number($('#flt_penalty').value || 0.35);
        payload.half_life=Number($('#flt_halflife').value || 7);
        log('log_filter', JSON.stringify(payload));
        const j=await postJSON('/api/filter', payload);
        log('log_filter', (j.code===0?'OK':'ERR')+"\\n"+(j.stdout||'')+"\\n"+(j.stderr||''));
      }
      async function runAll(){ await runFetch(); await runPrep(); await runTips(); await runFilter(); }

      function runFetchSSE(){ const qs = buildQS({provider:$('#fetch_provider').value,days:$('#fetch_days').value,out:$('#fetch_out').value}); sse('/api/stream/fetch?' + qs, 'log_fetch'); }
      function runPrepSSE(){ const qs = buildQS({src:$('#prep_src').value,out:$('#prep_out').value}); sse('/api/stream/prep?' + qs, 'log_prep'); }
      function runTipsSSE(){ const qs = buildQS({history:$('#tips_hist').value,fixtures:$('#tips_fx').value,config:$('#tips_cfg').value,model_path:$('#tips_model').value,out:$('#tips_out').value}); sse('/api/stream/tips?' + qs, 'log_tips'); }
      function runFilterSSE(){
        const qs = buildQS({
          src: $('#flt_src').value,
          out: $('#flt_out').value,
          news: $('#flt_news').value,
          min_prob: $('#flt_minprob').value || 0.60,
          penalty: $('#flt_penalty').value || 0.35,
          half_life: $('#flt_halflife').value || 7
        });
        sse('/api/stream/filter?' + qs, 'log_filter');
      }

      document.addEventListener('DOMContentLoaded', function(){
        document.getElementById('btnFetch').addEventListener('click', runFetch);
        document.getElementById('btnFetchSSE').addEventListener('click', runFetchSSE);
        document.getElementById('btnPrep').addEventListener('click', runPrep);
        document.getElementById('btnPrepSSE').addEventListener('click', runPrepSSE);
        document.getElementById('btnTips').addEventListener('click', runTips);
        document.getElementById('btnTipsSSE').addEventListener('click', runTipsSSE);
        document.getElementById('btnFilter').addEventListener('click', runFilter);
        document.getElementById('btnFilterSSE').addEventListener('click', runFilterSSE);
        document.getElementById('btnAll').addEventListener('click', runAll);
        document.getElementById('btnClear').addEventListener('click', function(){ L.textContent=''; });
        document.getElementById('btnSave').addEventListener('click', async function(){
          const obj={};
          ['fetch_provider','fetch_days','fetch_out','prep_src','prep_out',
           'tips_hist','tips_fx','tips_cfg','tips_model','tips_out',
           'flt_src','flt_out','flt_news','flt_minprob','flt_penalty','flt_halflife']
            .forEach(function(k){ const el=document.getElementById(k); if(el) obj[k]=el.value; });
          await fetch('/api/save_settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)});
          alert('Definições guardadas.');
        });
      });
    </script>
    """

    html = (html_tpl
      .replace("%%FETCH_PROVIDER%%", g('fetch_provider','sofascore_playwright'))
      .replace("%%FETCH_DAYS%%", g('fetch_days',2))
      .replace("%%FETCH_OUT%%", g('fetch_out', r'data\fixtures\latest.csv'))
      .replace("%%PREP_SRC%%", g('prep_src', r'data\fixtures\latest.csv'))
      .replace("%%PREP_OUT%%", g('prep_out', r'data\fixtures\latest_for_tips.csv'))
      .replace("%%TIPS_HIST%%", g('tips_hist','data/processed/matches.csv'))
      .replace("%%TIPS_FX%%", g('tips_fx','data/fixtures/latest_for_tips.csv'))
      .replace("%%TIPS_CFG%%", g('tips_cfg','configs/default.yaml'))
      .replace("%%TIPS_MODEL%%", g('tips_model','models/model.joblib'))
      .replace("%%TIPS_OUT%%", g('tips_out','outputs/tips.csv'))
      .replace("%%FLT_SRC%%", g('flt_src','outputs/tips.csv'))
      .replace("%%FLT_OUT%%", g('flt_out','outputs/tips_filtered.csv'))
      .replace("%%FLT_NEWS%%", g('flt_news', r'data\news\news_flags.csv'))
      .replace("%%FLT_MINPROB%%", g('flt_minprob', 0.60))
      .replace("%%FLT_PENALTY%%", g('flt_penalty', 0.35))
      .replace("%%FLT_HALFLIFE%%", g('flt_halflife', 7))
    )
    return layout(html, title="Pipeline")

# ---------- files ----------
@app.get("/files")
def page_files():
    html = """
    <div class=card>
      <div class="flex items-center justify-between mb-3">
        <h2 class="font-semibold text-lg">Ficheiros</h2>
        <button id=btnRefresh class="btn btn-ghost">Atualizar</button>
      </div>
      <div id=files class=text-sm>(a carregar…)</div>
    </div>
    <div class=card>
      <h3 class="font-semibold mb-2">Preview CSV</h3>
      <div class="grid grid-cols-1 gap-2 mb-2">
        <input id=pv_path class="border rounded-lg px-3 py-2" value="outputs/tips_filtered.csv" />
        <div class="flex items-center gap-2">
          <label class=text-xs>Linhas</label>
          <input id=pv_n type=number class="border rounded-lg px-3 py-2 w-24" value=50 />
          <button id=btnPreview class="btn btn-ghost">Preview</button>
        </div>
      </div>
      <div id=preview class="overflow-auto max-h-96 border rounded-xl"></div>
    </div>
    <div class=card>
      <h3 class="font-semibold mb-2">Upload</h3>
      <form id=upForm class="flex flex-col md:flex-row gap-3 items-start">
        <input id=up_file type=file class="border rounded-lg px-3 py-2" />
        <input id=up_dest class="border rounded-lg px-3 py-2 flex-1" placeholder="destino (ex.: configs/default.yaml ou models/model.joblib)" />
        <button class="btn btn-primary" type=submit>Enviar</button>
      </form>
      <div id=up_msg class=text-sm text-neutral-500></div>
    </div>
    <script>
      const $=(s)=>document.querySelector(s);
      async function refreshFiles(){
        const r=await fetch('/api/list'); const j=await r.json();
        const fmt=(ts)=> new Date(ts*1000).toLocaleString();
        function list(title, arr){
          if(!arr||!arr.length) return `<div class='text-xs text-neutral-500'>(sem ${title})</div>`;
          return `<h4 class='font-semibold mb-2'>${title}</h4><ul class='space-y-1'>` +
            arr.map(x=>`<li><a class='link' href='/download/${x.name}' target='_blank'>${x.name}</a> <span class='text-xs text-neutral-500'>${(x.size/1024).toFixed(1)} KB • ${fmt(x.mtime)}</span></li>`).join('') +
            `</ul>`;
        }
        document.getElementById('files').innerHTML =
          list('fixtures', j.fixtures) + list('outputs', j.outputs) + list('processed', j.processed) + list('news', j.news);
      }
      async function previewCsv(){
        const p=document.getElementById('pv_path').value; const n=Number(document.getElementById('pv_n').value||50);
        const r=await fetch(`/api/preview_csv?path=${encodeURIComponent(p)}&n=${n}`); const j=await r.json();
        if(j.error){ document.getElementById('preview').innerHTML = `<div class='text-red-600'>${j.error}</div>`; return; }
        const cols=j.columns||[]; const rows=j.rows||[];
        const th = `<tr>${cols.map(c=>`<th class='px-3 py-2 bg-neutral-50 dark:bg-neutral-800 sticky top-0 border-b'>${c}</th>`).join('')}</tr>`;
        const tb = rows.map(r=>`<tr>${r.map(cell=>`<td class='px-3 py-1 border-b'>${String(cell)}</td>`).join('')}</tr>`).join('');
        document.getElementById('preview').innerHTML = `<div class='overflow-auto max-h-96'><table class='min-w-full text-sm border'>${th}${tb}</table></div>`;
      }
      document.addEventListener('DOMContentLoaded', ()=>{
        refreshFiles(); document.getElementById('btnRefresh').addEventListener('click', refreshFiles);
        document.getElementById('btnPreview').addEventListener('click', previewCsv);
        document.getElementById('upForm').addEventListener('submit', async (e)=>{
          e.preventDefault();
          const f=document.getElementById('up_file').files[0]; const dest=document.getElementById('up_dest').value;
          if(!f||!dest){ document.getElementById('up_msg').textContent='Escolha o ficheiro e o destino.'; return;}
          const fd=new FormData(); fd.append('file', f); fd.append('dest', dest);
          const r=await fetch('/api/upload',{method:'POST', body:fd}); const j=await r.json();
          document.getElementById('up_msg').textContent = j.ok? ('Guardado em '+j.saved) : (j.error||'erro');
        });
      });
    </script>
    """
    return layout(html, title="Ficheiros")

# ---------- settings ----------
@app.get("/settings")
def page_settings():
    s = read_settings()
    def val(k, d=''): 
        return str(s.get(k, d)) if isinstance(s, dict) else str(d)

    html_tpl = r"""
    <div class=card>
      <h2 class="font-semibold text-lg mb-3">Definições</h2>
      <div class="grid md:grid-cols-2 gap-4">
        <div class=card>
          <h3 class="font-semibold mb-2">Defaults do Pipeline</h3>
          <div class="grid grid-cols-1 gap-2 text-sm">
            <label>Provider<input id=fetch_provider class="border rounded-lg px-3 py-2" value="%%P_FETCH_PROVIDER%%"/></label>
            <label>Dias<input id=fetch_days type=number class="border rounded-lg px-3 py-2" value="%%P_FETCH_DAYS%%"/></label>
            <label>Fetch out<input id=fetch_out class="border rounded-lg px-3 py-2" value="%%P_FETCH_OUT%%"/></label>
            <label>Prep src<input id=prep_src class="border rounded-lg px-3 py-2" value="%%P_PREP_SRC%%"/></label>
            <label>Prep out<input id=prep_out class="border rounded-lg px-3 py-2" value="%%P_PREP_OUT%%"/></label>
            <label>History<input id=tips_hist class="border rounded-lg px-3 py-2" value="%%P_TIPS_HIST%%"/></label>
            <label>Fixtures<input id=tips_fx class="border rounded-lg px-3 py-2" value="%%P_TIPS_FX%%"/></label>
            <label>Config<input id=tips_cfg class="border rounded-lg px-3 py-2" value="%%P_TIPS_CFG%%"/></label>
            <label>Model<input id=tips_model class="border rounded-lg px-3 py-2" value="%%P_TIPS_MODEL%%"/></label>
            <label>Tips out<input id=tips_out class="border rounded-lg px-3 py-2" value="%%P_TIPS_OUT%%"/></label>
            <label>Filter src<input id=flt_src class="border rounded-lg px-3 py-2" value="%%P_FLT_SRC%%"/></label>
            <label>Filter out<input id=flt_out class="border rounded-lg px-3 py-2" value="%%P_FLT_OUT%%"/></label>

            <label>News CSV<input id=flt_news class="border rounded-lg px-3 py-2" value="%%P_FLT_NEWS%%"/></label>
            <label>Min Prob<input id=flt_minprob type=number step="0.01" class="border rounded-lg px-3 py-2" value="%%P_FLT_MINPROB%%"/></label>
            <label>Penalty<input id=flt_penalty type=number step="0.01" class="border rounded-lg px-3 py-2" value="%%P_FLT_PENALTY%%"/></label>
            <label>Half-life<input id=flt_halflife type=number class="border rounded-lg px-3 py-2" value="%%P_FLT_HALFLIFE%%"/></label>
          </div>
          <div class="mt-3"><button id=btnSave class="btn btn-primary">Guardar</button></div>
        </div>
        <div class=card>
          <h3 class="font-semibold mb-2">Aparência</h3>
          <button id=themeBtn class="btn btn-ghost">🌓 Alternar tema</button>
          <p class="text-xs text-neutral-500 mt-2">Preferência é guardada no browser.</p>
        </div>
      </div>
    </div>
    <script>
      const $=(s)=>document.querySelector(s);
      document.addEventListener('DOMContentLoaded', function(){
        document.getElementById('btnSave').addEventListener('click', async function(){
          const keys=['fetch_provider','fetch_days','fetch_out','prep_src','prep_out','tips_hist','tips_fx','tips_cfg','tips_model','tips_out','flt_src','flt_out','flt_news','flt_minprob','flt_penalty','flt_halflife'];
          const obj={}; keys.forEach(function(k){ obj[k]=document.getElementById(k).value; });
          await fetch('/api/save_settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(obj)});
          alert('Guardado.');
        });
        document.getElementById('themeBtn').addEventListener('click', function(){
          const d=document.documentElement; const dark=d.classList.toggle('dark');
          localStorage.setItem('tt_theme', dark?'dark':'light');
        });
      });
    </script>
    """

    html = (html_tpl
      .replace("%%P_FETCH_PROVIDER%%", val('fetch_provider','sofascore_playwright'))
      .replace("%%P_FETCH_DAYS%%", val('fetch_days',2))
      .replace("%%P_FETCH_OUT%%", val('fetch_out', r'data\fixtures\latest.csv'))
      .replace("%%P_PREP_SRC%%", val('prep_src', r'data\fixtures\latest.csv'))
      .replace("%%P_PREP_OUT%%", val('prep_out', r'data\fixtures\latest_for_tips.csv'))
      .replace("%%P_TIPS_HIST%%", val('tips_hist','data/processed/matches.csv'))
      .replace("%%P_TIPS_FX%%", val('tips_fx','data/fixtures/latest_for_tips.csv'))
      .replace("%%P_TIPS_CFG%%", val('tips_cfg','configs/default.yaml'))
      .replace("%%P_TIPS_MODEL%%", val('tips_model','models/model.joblib'))
      .replace("%%P_TIPS_OUT%%", val('tips_out','outputs/tips.csv'))
      .replace("%%P_FLT_SRC%%", val('flt_src','outputs/tips.csv'))
      .replace("%%P_FLT_OUT%%", val('flt_out','outputs/tips_filtered.csv'))
      .replace("%%P_FLT_NEWS%%", val('flt_news', r'data\news\news_flags.csv'))
      .replace("%%P_FLT_MINPROB%%", val('flt_minprob', 0.60))
      .replace("%%P_FLT_PENALTY%%", val('flt_penalty', 0.35))
      .replace("%%P_FLT_HALFLIFE%%", val('flt_halflife', 7))
    )
    return layout(html, title="Definições")

# ---------- about ----------
@app.get("/about")
def page_about():
    html = """
    <div class=card>
      <h2 class="font-semibold text-lg mb-2">Sobre</h2>
      <p class=text-sm>Dashboard profissional para gerir a pipeline Tennis Tipster (fixtures → prep → tips → filter). Construído com Flask, Tailwind e SSE para logs em tempo real.</p>
      <ul class="list-disc pl-6 text-sm mt-3">
        <li>Execução por passos ou completa</li>
        <li>Logs em tempo real (SSE) e logs consolidados</li>
        <li>Gestão de ficheiros, preview e download</li>
        <li>Definições persistidas em <code>configs/ui_settings.json</code></li>
      </ul>
    </div>
    """
    return layout(html, title="Sobre")

# ----------------------- run -----------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
