"""Launch the separate web app with one command; leaves Streamlit untouched."""
import os,shutil,subprocess,sys
from pathlib import Path
WEB=Path(__file__).resolve().parent
bundled=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
node=shutil.which('node') or (str(bundled) if bundled.exists() else None)
if not node:raise SystemExit('Node.js is missing. Install Node.js 20.19+ or 22.12+, then run the setup commands in web/README.md.')
if not (WEB/'node_modules/tsx/dist/cli.mjs').exists():raise SystemExit('Install local dependencies first: cd web && npm install')
env={**os.environ,'PATH':str(Path(node).parent)+os.pathsep+os.environ.get('PATH','')}
processes=[]
try:
 for args in [['node_modules/tsx/dist/cli.mjs','server/index.ts'],['node_modules/vite/bin/vite.js','--config','client/vite.config.ts']]:
  processes.append(subprocess.Popen([node,*args],cwd=WEB,env=env))
 print('\nOpen http://127.0.0.1:5173 — Ctrl+C stops both processes.\n',flush=True)
 while True:
  for p in processes:
   try:p.wait(timeout=.5)
   except subprocess.TimeoutExpired:continue
   raise SystemExit(f'Web process exited with status {p.returncode}; check the port or preceding error.')
except KeyboardInterrupt:pass
finally:
 for p in processes:p.terminate()
 for p in processes:
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:p.kill()
