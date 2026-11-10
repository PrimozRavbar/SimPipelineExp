"""FastAPI app around one SimSession. The UI (static/index.html) talks to
it only through the JSON endpoints below.

GET  /api/state                  clock, speed, serving model, budget, references
GET  /api/metrics?since=T        per-tick shown/clicks/ctr/active users/latency for ticks > T
GET  /api/events?since=SEQ       event log entries with seq > SEQ
GET  /api/models                 model versions, training stats, rollback target
GET  /api/pipeline               component status + editable settings
GET  /api/incidents              incidents opened by the alert rule
POST /api/play | /api/pause      start or stop the clock
POST /api/speed   {"speed": 1|10|100}
POST /api/step    {"ticks": n}   advance n ticks now (only while paused)
POST /api/retrain | /api/rollback | /api/spark      queued for the next tick boundary
POST /api/settings {name: value, ...}               queued for the next tick boundary
GET  /api/faults                 faults so far (hidden chaos faults only counted while active)
POST /api/fault {"kind": k, "duration": n|null}     inject a fault drill at the next tick
POST /api/repair/{action}        restart_flink | rebuild_index | restart_model_server
POST /api/canary/promote | /api/canary/abort       end a running canary
POST /api/models/{v}/export      save model v (towers, index, ranker) under <run_dir>/models/v<v>
"""

from pathlib import Path

from console.session import ActionError, SimSession

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(session):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse

    app = FastAPI(title="SimPipeline Console")
    app.state.session = session

    def run(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ActionError as error:
            raise HTTPException(status_code=400, detail=str(error))

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state():
        return session.state()

    @app.get("/api/metrics")
    def metrics(since: int = 0):
        return session.metrics(since)

    @app.get("/api/events")
    def events(since: int = 0):
        return {"events": session.events(since)}

    @app.get("/api/models")
    def models():
        return session.models()

    @app.get("/api/pipeline")
    def pipeline():
        return session.pipeline()

    @app.get("/api/incidents")
    def incidents():
        return {"incidents": session.incidents_view()}

    @app.post("/api/play")
    def play():
        session.play()
        return session.state()

    @app.post("/api/pause")
    def pause():
        session.pause()
        return session.state()

    @app.post("/api/speed")
    def speed(body: dict = Body(...)):
        run(session.set_speed, body.get("speed"))
        return session.state()

    @app.post("/api/step")
    def step(body: dict = Body(default={})):
        if session.playing:
            raise HTTPException(status_code=409, detail="pause the clock before stepping")
        ticks = int(body.get("ticks", 1))
        if not 1 <= ticks <= 1000:
            raise HTTPException(status_code=400, detail="ticks must be 1..1000")
        session.step(ticks)
        return session.state()

    def action(name, **params):
        record = run(session.submit, name, **params)
        return {"action": {k: record[k] for k in ("id", "name", "status", "submitted_tick")},
                "applies_at_tick": session.tick}

    @app.post("/api/retrain")
    def retrain():
        return action("retrain")

    @app.post("/api/rollback")
    def rollback():
        return action("rollback")

    @app.post("/api/spark")
    def spark():
        return action("spark")

    @app.post("/api/settings")
    def settings(body: dict = Body(...)):
        return action("settings", **body)

    @app.post("/api/models/{version}/export")
    def export(version: int):
        return action("export", version=version)

    @app.post("/api/canary/promote")
    def canary_promote():
        return action("promote_canary")

    @app.post("/api/canary/abort")
    def canary_abort():
        return action("abort_canary")

    @app.get("/api/faults")
    def faults():
        return session.faults_view()

    @app.post("/api/fault")
    def fault(body: dict = Body(...)):
        return action("fault", kind=body.get("kind"), duration=body.get("duration"))

    @app.post("/api/repair/{name}")
    def repair(name: str):
        if name not in ("restart_flink", "rebuild_index", "restart_model_server"):
            raise HTTPException(status_code=404, detail="unknown repair %r" % name)
        return action(name)

    return app


def serve(host="127.0.0.1", port=8000, seed=42, users_per_tick=10, runs_dir="runs", world="static"):
    import faulthandler
    import signal

    import uvicorn

    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1)   # kill -USR1 <pid> dumps every thread's stack

    print("Building the simulation (seed %d, %s world)..." % (seed, world), flush=True)
    session = SimSession(seed=seed, users_per_tick=users_per_tick, runs_dir=runs_dir, world=world)
    session.start()
    app = create_app(session)
    shown_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print("SimPipeline console: http://%s:%d" % (shown_host, port), flush=True)
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        session.close()
