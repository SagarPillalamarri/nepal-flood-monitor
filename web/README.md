# Nepal Flood Monitor — minimal web frontend

A separate React interface for the accepted Bagmati project. Streamlit remains the reference implementation and working fallback. No existing Python pipeline, model, feature, threshold or data file was changed to build this frontend.

## Start on this machine

From the project root:

```sh
venv/bin/python web/run.py
```

Open **http://127.0.0.1:5173**. Ctrl+C stops both the frontend and its API. The launcher uses an existing Node executable, including the installed Codex runtime if Node is absent from PATH. There are no global package installs.

## Fresh setup

Requires the project's Python environment plus Node.js 20.19+, 22.12+ or a newer supported version. This build was tested using the bundled Node 24 runtime.

```sh
cd web
npm install
npm run dev:server
```

In another terminal:

```sh
cd web
npm run dev:client
```

Production build and single-server local preview:

```sh
cd web
npm run build
npm start
```

Open http://127.0.0.1:8787. The server binds to loopback only. This is a local research dashboard, not an externally deployed warning service.

## Experience

- **Live monitor:** four essential readings, one selectable chart, frozen operational score bars and expandable explanations. The stale measured gauge remains prominently flagged. Refresh data requests weather/flow and DHM through the existing Python functions; partial failures keep saved data and report the error.
- **September replay:** previous/next/play/date slider, saved research scores and synchronized rainfall, moisture and river-flow timelines. Ongoing flood days remain unscored.
- **Data pipeline:** saved lake metrics and scaling chart, with one short coverage note. Raw parity tables, SQL results and the Spark physical plan remain in the research reports and are omitted from the web interface.
- Light/dark appearance, responsive laptop/mobile layout, keyboard focus states and reduced-motion support. Chart units and source captions are retained. The signal summary is a compact five-row view rather than a transcript of the Python panel. Full scientific evidence remains in the project reports.

## Architecture and invariants

`client/`: React + Vite + TypeScript, React Router, Recharts, Tailwind and CSS. Design tokens live in one file, `client/src/theme.css`; layout rules are in `styles.css`.

`server/index.ts`: Express + TypeScript. Read-only GET endpoints invoke `server/read_project.py` against saved CSV/JSON/SQLite-backed project outputs. Python is reused for the existing precursor explanation; Node does not calculate features or model scores. NaNs become JSON null; numeric precision is preserved. Every API response contains source filenames and generated_at.

- GET `/api/live`
- GET `/api/replay?start=2024-09-20&end=2024-09-30`
- GET `/api/pipeline`
- POST `/api/live/fetch` invokes the existing dual-source refresh. Only one refresh is accepted at a time. Partial source errors return HTTP 207 and remain visible. No automatic polling occurs.

The API reads local saved snapshots, not browser credentials or API keys. It can show missing-data and unavailable-service states. A fresh Git clone lacks the intentionally excluded live database/snapshots; run the existing Python pollers, or use Refresh data. The UI does not enable disabled analogue resemblance labels or alter either set of model results.

## Verification

Start the app, then run from the project root:

```sh
venv/bin/python web/check_parity.py
```

The check recursively compares all API data with source-reader values, independently checks CSV floats for replay/live records, verifies the SQLite-backed operational record and frozen model hashes. Only time-of-request fields are excluded from exact equality. Refresh is not invoked during parity checking.

`parity_output.txt` records the result. Browser review screenshots are in `screenshots/` for each page in light and dark appearance. Screenshots reflect the in-app browser viewport. The refresh button was not used during browser QA, to avoid changing saved scientific inputs simply for a visual review.
