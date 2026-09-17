"""
MLB Sim API Server
==================
Serves Monte Carlo-computed PA outcome distributions to the browser UI.

Run:  python sim_api.py
Then open the artifact — it will fetch from http://localhost:5001/api/...

Endpoints:
  GET /api/game          — full game setup (lineups, pitchers, park, weather)
  GET /api/pa_dist       — PA outcome distribution for batter vs pitcher
  POST /api/simulate     — run a full N-sim Monte Carlo, return aggregated results
  GET /api/live_game     — stream a single simulated game event-by-event (SSE)
"""

import json
import random
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict
from copy import copy

from flask import Flask, jsonify, request, Response, stream_with_context

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/", methods=["GET", "OPTIONS"])
def index():
    return jsonify({"status": "ok", "service": "ReelMLB Sim API", "endpoints": ["/api/status", "/api/game", "/api/pa_dist", "/api/live_game", "/api/simulate"]})

# ─────────────────────────────────────────────
# PARK FACTORS
# ─────────────────────────────────────────────
PARK_HR_FACTORS = {
    "COL":1.32,"CIN":1.18,"PHI":1.14,"NYY":1.12,"BOS":1.10,"HOU":1.08,
    "MIL":1.07,"ATL":1.06,"TEX":1.05,"ARI":1.04,"CHC":1.03,"CLE":1.02,
    "STL":1.01,"WSH":1.01,"TOR":1.00,"NYM":1.00,"LAA":0.99,"BAL":0.99,
    "DET":0.98,"MIN":0.98,"CWS":0.97,"KC":0.97,"TB":0.96,"SEA":0.95,
    "MIA":0.94,"OAK":0.93,"PIT":0.92,"SF":0.91,"LAD":0.90,"SD":0.89,
}

# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────
@dataclass
class Batter:
    name: str
    team: str
    bats: str = "R"
    bb_rate: float = 0.085
    k_rate: float = 0.220
    barrel_rate: float = 0.070
    iso: float = 0.165
    babip: float = 0.295
    avg_exit_velo: float = 88.5
    avg_launch_angle: float = 12.0
    recent_form: float = 0.0

@dataclass
class Pitcher:
    name: str
    team: str
    throws: str = "R"
    era: float = 4.20
    stuff_quality: float = 50.0
    k_rate_bonus: float = 0.05
    bb_allowed_rate: float = 0.08
    hr_suppression: float = 0.0
    gb_rate: float = 0.45
    max_effective_pitches: int = 95
    pitch_count: int = 0

    @property
    def fatigue_factor(self):
        if self.pitch_count < 75: return 1.0
        if self.pitch_count < 95: return 1.0 + (self.pitch_count - 75) * 0.008
        return 1.0 + (self.pitch_count - 75) * 0.018

    def add_pitches(self, n=4): self.pitch_count += n


def weather_mult(temp_f, wind_out_mph):
    return max(0.5, (1 + (temp_f - 72) * 0.004) * (1 + wind_out_mph * 0.012))

def has_platoon(batter, pitcher):
    if batter.bats == "S": return True
    return (batter.bats == "L" and pitcher.throws == "R") or \
           (batter.bats == "R" and pitcher.throws == "L")

def hr_prob_pa(batter, pitcher, park_f, w_mult):
    base = batter.iso / 4.5
    barrel_adj = 1 + (batter.barrel_rate - 0.065) * 4.0
    platoon = 1.10 if has_platoon(batter, pitcher) else 1.0
    pitcher_adj = 1.0 - (pitcher.hr_suppression * 0.4)
    stuff_adj = 1.0 - (pitcher.stuff_quality - 50) * 0.002
    form_adj = 1.0 + batter.recent_form * 0.08
    fat = pitcher.fatigue_factor
    return max(0, min(0.15,
        base * barrel_adj * platoon * pitcher_adj * stuff_adj * form_adj * park_f * w_mult * fat
    ))

def pa_outcome_dist(batter, pitcher, park_f=1.0, w_mult=1.0):
    hr = hr_prob_pa(batter, pitcher, park_f, w_mult)
    bb = max(0.03, batter.bb_rate - pitcher.bb_allowed_rate * 0.3)
    k  = min(0.45, batter.k_rate  + pitcher.k_rate_bonus * 0.5 * pitcher.fatigue_factor)
    bip = max(0, 1 - hr - bb - k)
    triple = bip * 0.008
    double = bip * 0.095
    single = bip * batter.babip * 0.65
    out    = max(0, bip - triple - double - single)
    total  = hr + bb + k + triple + double + single + out
    return {
        "HR":  round(hr/total, 4),
        "3B":  round(triple/total, 4),
        "2B":  round(double/total, 4),
        "1B":  round(single/total, 4),
        "BB":  round(bb/total, 4),
        "K":   round(k/total, 4),
        "Out": round(out/total, 4),
        "hr_prob_raw": round(hr, 4),
    }

def roll_outcome(dist):
    r = random.random()
    cum = 0
    for ev, p in dist.items():
        if ev == "hr_prob_raw": continue
        cum += p
        if r <= cum: return ev
    return "Out"

# ─────────────────────────────────────────────
# GAME CONFIG
# Build from real data here — replace with pybaseball pull
# ─────────────────────────────────────────────
GAME_CONFIG = {
    "away_team": "NYY",
    "home_team": "BOS",
    "park": "BOS",
    "temp_f": 72.0,
    "wind_out_mph": 5.0,
    "date": "2026-09-17",

    "away_lineup": [
        {"name":"Aaron Judge",       "team":"NYY","bats":"R","barrel_rate":0.198,"iso":0.360,"bb_rate":0.182,"k_rate":0.245,"avg_exit_velo":97.2,"recent_form":0.5},
        {"name":"Juan Soto",         "team":"NYY","bats":"L","barrel_rate":0.135,"iso":0.260,"bb_rate":0.195,"k_rate":0.170,"avg_exit_velo":93.5,"recent_form":0.3},
        {"name":"Giancarlo Stanton", "team":"NYY","bats":"R","barrel_rate":0.155,"iso":0.310,"bb_rate":0.115,"k_rate":0.290,"avg_exit_velo":96.5,"recent_form":-0.1},
        {"name":"Anthony Rizzo",     "team":"NYY","bats":"L","barrel_rate":0.078,"iso":0.165,"bb_rate":0.095,"k_rate":0.195,"avg_exit_velo":88.5,"recent_form":0.0},
        {"name":"Gleyber Torres",    "team":"NYY","bats":"R","barrel_rate":0.065,"iso":0.145,"bb_rate":0.085,"k_rate":0.195,"avg_exit_velo":87.5,"recent_form":0.1},
        {"name":"Jazz Chisholm Jr.", "team":"NYY","bats":"L","barrel_rate":0.095,"iso":0.195,"bb_rate":0.075,"k_rate":0.285,"avg_exit_velo":91.2,"recent_form":0.2},
        {"name":"Oswaldo Cabrera",   "team":"NYY","bats":"S","barrel_rate":0.058,"iso":0.130,"bb_rate":0.065,"k_rate":0.215,"avg_exit_velo":86.8,"recent_form":-0.1},
        {"name":"Jose Trevino",      "team":"NYY","bats":"R","barrel_rate":0.045,"iso":0.095,"bb_rate":0.055,"k_rate":0.215,"avg_exit_velo":85.5,"recent_form":-0.2},
        {"name":"Anthony Volpe",     "team":"NYY","bats":"R","barrel_rate":0.055,"iso":0.120,"bb_rate":0.070,"k_rate":0.220,"avg_exit_velo":86.2,"recent_form":0.2},
    ],
    "home_lineup": [
        {"name":"Rafael Devers",    "team":"BOS","bats":"L","barrel_rate":0.112,"iso":0.240,"bb_rate":0.088,"k_rate":0.215,"avg_exit_velo":93.1,"recent_form":0.3},
        {"name":"Triston Casas",    "team":"BOS","bats":"L","barrel_rate":0.095,"iso":0.200,"bb_rate":0.120,"k_rate":0.250,"avg_exit_velo":91.8,"recent_form":0.1},
        {"name":"Jarren Duran",     "team":"BOS","bats":"L","barrel_rate":0.068,"iso":0.155,"bb_rate":0.070,"k_rate":0.210,"avg_exit_velo":89.5,"recent_form":0.4},
        {"name":"Tyler O'Neill",    "team":"BOS","bats":"R","barrel_rate":0.105,"iso":0.230,"bb_rate":0.065,"k_rate":0.280,"avg_exit_velo":92.4,"recent_form":0.0},
        {"name":"Rob Refsnyder",    "team":"BOS","bats":"R","barrel_rate":0.060,"iso":0.130,"bb_rate":0.095,"k_rate":0.175,"avg_exit_velo":87.2,"recent_form":-0.1},
        {"name":"Masataka Yoshida", "team":"BOS","bats":"L","barrel_rate":0.075,"iso":0.165,"bb_rate":0.110,"k_rate":0.135,"avg_exit_velo":88.8,"recent_form":0.2},
        {"name":"Ceddanne Rafaela", "team":"BOS","bats":"R","barrel_rate":0.045,"iso":0.110,"bb_rate":0.040,"k_rate":0.225,"avg_exit_velo":86.0,"recent_form":-0.2},
        {"name":"David Hamilton",   "team":"BOS","bats":"R","barrel_rate":0.035,"iso":0.085,"bb_rate":0.060,"k_rate":0.240,"avg_exit_velo":84.5,"recent_form":0.0},
        {"name":"Connor Wong",      "team":"BOS","bats":"R","barrel_rate":0.055,"iso":0.125,"bb_rate":0.055,"k_rate":0.265,"avg_exit_velo":87.0,"recent_form":0.1},
    ],

    "away_starter":  {"name":"Gerrit Cole",  "team":"NYY","throws":"R","era":2.95,"stuff_quality":80,"k_rate_bonus":0.14,"hr_suppression":0.25,"gb_rate":0.40,"max_effective_pitches":100},
    "home_starter":  {"name":"Brayan Bello", "team":"BOS","throws":"R","era":3.85,"stuff_quality":62,"k_rate_bonus":0.08,"hr_suppression":0.15,"gb_rate":0.48,"max_effective_pitches":92},
    "away_bullpen": [
        {"name":"Clay Holmes",    "team":"NYY","throws":"R","era":3.10,"stuff_quality":58,"k_rate_bonus":0.09,"hr_suppression":0.15,"gb_rate":0.62,"max_effective_pitches":30},
        {"name":"J. Loaisiga",   "team":"NYY","throws":"R","era":3.50,"stuff_quality":55,"k_rate_bonus":0.08,"hr_suppression":0.08,"gb_rate":0.50,"max_effective_pitches":25},
    ],
    "home_bullpen": [
        {"name":"Kenley Jansen", "team":"BOS","throws":"R","era":3.40,"stuff_quality":60,"k_rate_bonus":0.10,"hr_suppression":0.10,"gb_rate":0.40,"max_effective_pitches":30},
        {"name":"Chris Martin",  "team":"BOS","throws":"R","era":3.90,"stuff_quality":50,"k_rate_bonus":0.06,"hr_suppression":0.05,"gb_rate":0.45,"max_effective_pitches":25},
    ],
}


def make_batter(d): return Batter(**{k:v for k,v in d.items() if k in Batter.__dataclass_fields__})
def make_pitcher(d): return Pitcher(**{k:v for k,v in d.items() if k in Pitcher.__dataclass_fields__})


# ─────────────────────────────────────────────
# MONTE CARLO (full game sim for aggregates)
# ─────────────────────────────────────────────
class GameSim:
    def __init__(self, cfg):
        self.cfg = cfg
        self.park_f = PARK_HR_FACTORS.get(cfg["park"], 1.0)
        self.w_mult = weather_mult(cfg["temp_f"], cfg["wind_out_mph"])

    def _fresh_pitchers(self):
        away_sp = make_pitcher(self.cfg["away_starter"])
        home_sp = make_pitcher(self.cfg["home_starter"])
        away_bp = [make_pitcher(p) for p in self.cfg["away_bullpen"]]
        home_bp = [make_pitcher(p) for p in self.cfg["home_bullpen"]]
        return away_sp, home_sp, away_bp, home_bp

    def sim_one_game(self):
        away_sp, home_sp, away_bp, home_bp = self._fresh_pitchers()
        home_pitcher  = away_sp  # home team batters face away starter
        away_pitcher  = home_sp  # away team batters face home starter
        home_bp_idx = away_bp_idx = 0
        home_score = away_score = 0
        away_idx = home_idx = 0
        home_runs = []
        linescore = {"away": [], "home": []}

        def active_p(side):
            if side == "home":
                return home_pitcher if home_bp_idx == 0 else away_bp[home_bp_idx-1]
            else:
                return away_pitcher if away_bp_idx == 0 else home_bp[away_bp_idx-1]

        def maybe_change(half, inning, pc):
            nonlocal home_bp_idx, away_bp_idx, home_pitcher, away_pitcher
            side = "home" if half == "top" else "away"
            p = active_p(side)
            bp = away_bp if side == "home" else home_bp
            idx_key = "home" if side == "home" else "away"
            cur_idx = home_bp_idx if side == "home" else away_bp_idx
            if (pc >= p.max_effective_pitches + 15 or
                    (inning >= 6 and pc >= 80 and p.era > 4.8)) and cur_idx < len(bp):
                if side == "home": home_bp_idx += 1
                else: away_bp_idx += 1
                p.pitch_count = 0

        for inning in range(1, 10):
            for half in ("top", "bottom"):
                if inning == 9 and half == "bottom" and home_score > away_score:
                    break
                lineup = self.cfg["away_lineup"] if half == "top" else self.cfg["home_lineup"]
                side = "home" if half == "top" else "away"
                p = active_p(side)
                outs = 0; runs = 0
                bases = [None, None, None]
                pc_at_start = p.pitch_count

                while outs < 3:
                    if half == "top":
                        batter_d = lineup[away_idx % 9]; away_idx += 1
                    else:
                        batter_d = lineup[home_idx % 9]; home_idx += 1
                    batter = make_batter(batter_d)
                    maybe_change(half, inning, p.pitch_count)
                    p = active_p(side)
                    dist = pa_outcome_dist(batter, p, self.park_f, self.w_mult)
                    outcome = roll_outcome(dist)
                    p.add_pitches(4 if outcome in ("BB","K") else 3)

                    scored = 0
                    if outcome == "HR":
                        scored = 1 + sum(1 for b in bases if b)
                        bases = [None,None,None]
                        home_runs.append({"batter":batter.name,"team":self.cfg["away_team" if half=="top" else "home_team"],"inning":inning,"half":half})
                    elif outcome == "3B":
                        scored = sum(1 for b in bases if b)
                        bases = [None, None, batter.name]
                    elif outcome == "2B":
                        scored = (1 if bases[2] else 0) + (1 if bases[1] else 0)
                        bases = [None, batter.name, bases[0]]
                    elif outcome == "1B":
                        scored += 1 if bases[2] else 0
                        scored += 1 if (bases[1] and random.random()<0.55) else 0
                        bases = [batter.name, bases[0], None]
                    elif outcome == "BB":
                        if all(bases): scored = 1
                        if bases[0] and bases[1]: bases[2] = bases[1]
                        if bases[0]: bases[1] = bases[0]
                        bases[0] = batter.name
                    else:
                        outs += 1

                    runs += scored

                if half == "top": away_score += runs; linescore["away"].append(runs)
                else:             home_score += runs; linescore["home"].append(runs)

        # Extra innings
        extra = 10
        while away_score == home_score and extra <= 13:
            for half in ("top","bottom"):
                lineup = self.cfg["away_lineup"] if half=="top" else self.cfg["home_lineup"]
                side = "home" if half=="top" else "away"
                p = active_p(side)
                outs = 0; runs = 0; bases = [None,None,None]
                while outs < 3:
                    if half=="top":
                        batter_d = lineup[away_idx%9]; away_idx+=1
                    else:
                        batter_d = lineup[home_idx%9]; home_idx+=1
                    batter = make_batter(batter_d)
                    dist = pa_outcome_dist(batter, p, self.park_f, self.w_mult)
                    outcome = roll_outcome(dist)
                    p.add_pitches(4 if outcome in ("BB","K") else 3)
                    if outcome in ("K","Out"): outs+=1
                    elif outcome=="HR":
                        runs += 1+sum(1 for b in bases if b); bases=[None,None,None]
                    elif outcome=="1B":
                        runs += 1 if bases[2] else 0; bases=[batter.name,bases[0],None]
                    elif outcome in ("2B","3B","BB"): pass
                if half=="top": away_score+=runs; linescore["away"].append(runs)
                else:           home_score+=runs; linescore["home"].append(runs)
            extra+=1

        return {
            "away_score": away_score, "home_score": home_score,
            "winner": self.cfg["away_team"] if away_score > home_score else self.cfg["home_team"],
            "home_runs": home_runs, "linescore": linescore,
        }

    def monte_carlo(self, n=5000):
        results = []
        hr_counter = defaultdict(int)
        score_hist = defaultdict(int)
        for _ in range(n):
            r = self.sim_one_game()
            results.append(r)
            score_hist[f"{r['away_score']}-{r['home_score']}"] += 1
            for hr in r["home_runs"]: hr_counter[hr["batter"]] += 1

        away_wins = sum(1 for r in results if r["winner"] == self.cfg["away_team"])
        avg_away = sum(r["away_score"] for r in results) / n
        avg_home = sum(r["home_score"] for r in results) / n
        avg_hrs  = sum(len(r["home_runs"]) for r in results) / n

        return {
            "simulations": n,
            "win_probability": {
                self.cfg["away_team"]: round(away_wins/n, 3),
                self.cfg["home_team"]: round(1 - away_wins/n, 3),
            },
            "projected_score": {
                self.cfg["away_team"]: round(avg_away, 2),
                self.cfg["home_team"]: round(avg_home, 2),
            },
            "avg_home_runs": round(avg_hrs, 2),
            "hr_probabilities": {
                k: round(v/n, 3) for k,v in sorted(hr_counter.items(), key=lambda x:-x[1])
            },
            "top_scores": sorted(score_hist.items(), key=lambda x:-x[1])[:8],
            "park": self.cfg["park"],
            "park_factor": PARK_HR_FACTORS.get(self.cfg["park"], 1.0),
            "weather_mult": round(weather_mult(self.cfg["temp_f"], self.cfg["wind_out_mph"]), 3),
        }


# ─────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────

@app.route("/api/game")
def get_game():
    """Return full game config (lineups, pitchers, park, weather)."""
    park_f = PARK_HR_FACTORS.get(GAME_CONFIG["park"], 1.0)
    wm = weather_mult(GAME_CONFIG["temp_f"], GAME_CONFIG["wind_out_mph"])
    return jsonify({**GAME_CONFIG, "park_factor": park_f, "weather_mult": round(wm,3)})


@app.route("/api/pa_dist")
def get_pa_dist():
    """
    Compute PA outcome distribution for a batter vs pitcher.
    Query params:
      batter_name  — must match a name in the lineup
      pitcher_name — must match a pitcher name
      pitch_count  — current pitcher pitch count (for fatigue)
    """
    batter_name  = request.args.get("batter_name", "")
    pitcher_name = request.args.get("pitcher_name", "")
    pitch_count  = int(request.args.get("pitch_count", 0))

    # Find batter
    all_batters = GAME_CONFIG["away_lineup"] + GAME_CONFIG["home_lineup"]
    batter_d = next((b for b in all_batters if b["name"] == batter_name), None)
    if not batter_d:
        return jsonify({"error": f"Batter '{batter_name}' not found"}), 404

    # Find pitcher (starter or bullpen)
    all_pitchers = [
        GAME_CONFIG["away_starter"], GAME_CONFIG["home_starter"],
        *GAME_CONFIG["away_bullpen"], *GAME_CONFIG["home_bullpen"]
    ]
    pitcher_d = next((p for p in all_pitchers if p["name"] == pitcher_name), None)
    if not pitcher_d:
        return jsonify({"error": f"Pitcher '{pitcher_name}' not found"}), 404

    batter  = make_batter(batter_d)
    pitcher = make_pitcher(pitcher_d)
    pitcher.pitch_count = pitch_count

    park_f = PARK_HR_FACTORS.get(GAME_CONFIG["park"], 1.0)
    wm     = weather_mult(GAME_CONFIG["temp_f"], GAME_CONFIG["wind_out_mph"])

    dist = pa_outcome_dist(batter, pitcher, park_f, wm)
    dist["batter"] = batter.name
    dist["pitcher"] = pitcher.name
    dist["fatigue_factor"] = round(pitcher.fatigue_factor, 3)
    dist["platoon_advantage"] = has_platoon(batter, pitcher)
    dist["park_factor"] = park_f
    dist["weather_mult"] = round(wm, 3)

    return jsonify(dist)


@app.route("/api/simulate", methods=["POST"])
def run_simulation():
    """Run a full Monte Carlo simulation. Body: {"n": 5000}"""
    body = request.get_json(silent=True) or {}
    n = min(int(body.get("n", 5000)), 25000)
    sim = GameSim(GAME_CONFIG)
    result = sim.monte_carlo(n)
    return jsonify(result)


@app.route("/api/live_game")
def live_game_stream():
    """
    Server-Sent Events stream of a single game simulation.
    Each PA emits one JSON event so the UI can animate it live.

    ?delay=0.8  — seconds between events (default 0.8, min 0.05)
    """
    delay = max(0.05, float(request.args.get("delay", 0.8)))

    def generate():
        cfg = GAME_CONFIG
        park_f = PARK_HR_FACTORS.get(cfg["park"], 1.0)
        wm = weather_mult(cfg["temp_f"], cfg["wind_out_mph"])

        away_sp = make_pitcher(cfg["away_starter"])
        home_sp = make_pitcher(cfg["home_starter"])
        away_bp = [make_pitcher(p) for p in cfg["away_bullpen"]]
        home_bp = [make_pitcher(p) for p in cfg["home_bullpen"]]
        home_pitcher_obj = away_sp
        away_pitcher_obj = home_sp
        home_bp_idx = away_bp_idx = 0

        home_score = away_score = 0
        away_idx = home_idx = 0
        linescore = {"away": [], "home": []}

        def active_p(side):
            if side == "home": return away_sp if home_bp_idx==0 else away_bp[home_bp_idx-1]
            else:              return home_sp if away_bp_idx==0 else home_bp[away_bp_idx-1]

        def emit(event_type, data):
            return f"data: {json.dumps({'type': event_type, **data})}\n\n"

        # Send 2KB padding comment to force Railway/nginx to flush the buffer immediately
        yield ": " + " " * 2048 + "\n\n"

        # Emit game setup
        yield emit("game_start", {
            "away_team": cfg["away_team"], "home_team": cfg["home_team"],
            "away_starter": cfg["away_starter"]["name"],
            "home_starter": cfg["home_starter"]["name"],
            "park": cfg["park"], "park_factor": park_f,
            "weather_mult": round(wm,3),
            "away_lineup": [b["name"] for b in cfg["away_lineup"]],
            "home_lineup": [b["name"] for b in cfg["home_lineup"]],
        })
        time.sleep(delay * 0.5)

        for inning in range(1, 14):
            for half in ("top", "bottom"):
                if inning > 9 and away_score != home_score: break
                if inning == 9 and half == "bottom" and home_score > away_score: break
                if inning > 13: break

                lineup = cfg["away_lineup"] if half=="top" else cfg["home_lineup"]
                side = "home" if half=="top" else "away"
                p = active_p(side)

                yield emit("half_inning_start", {
                    "inning": inning, "half": half,
                    "pitcher": p.name, "pitch_count": p.pitch_count,
                    "away_score": away_score, "home_score": home_score,
                })
                time.sleep(delay * 0.4)

                outs = 0; runs_this_half = 0
                bases = [None, None, None]

                while outs < 3:
                    # Maybe change pitcher
                    if (p.pitch_count >= p.max_effective_pitches+15 or
                            (inning>=6 and p.pitch_count>=80 and p.era>4.8)):
                        bp = away_bp if side=="home" else home_bp
                        idx = home_bp_idx if side=="home" else away_bp_idx
                        if idx < len(bp):
                            if side=="home": home_bp_idx+=1
                            else:            away_bp_idx+=1
                            p = active_p(side)
                            p.pitch_count = 0
                            yield emit("pitching_change", {
                                "inning": inning, "half": half,
                                "new_pitcher": p.name,
                            })
                            time.sleep(delay * 0.6)

                    if half=="top":
                        batter_d = lineup[away_idx%9]; away_idx+=1
                    else:
                        batter_d = lineup[home_idx%9]; home_idx+=1

                    batter = make_batter(batter_d)
                    dist = pa_outcome_dist(batter, p, park_f, wm)
                    outcome = roll_outcome(dist)
                    p.add_pitches(4 if outcome in ("BB","K") else 3)

                    # Statcast sim
                    ev_val = la_val = dist_val = None
                    if outcome not in ("K","BB","Out"):
                        ev_val  = round(batter.avg_exit_velo + (random.random()-0.5)*6, 1)
                        la_val  = round({"HR":30,"3B":15,"2B":12,"1B":9}.get(outcome,5) + (random.random()-0.5)*10, 1)
                        dist_val= int({"HR":400,"3B":290,"2B":250,"1B":160}.get(outcome,80) + random.random()*60)
                    elif outcome=="Out":
                        ev_val = round(70 + random.random()*20, 1)
                        la_val = round(-5 + random.random()*20, 1)
                        dist_val = int(60 + random.random()*120)

                    scored = 0
                    if outcome=="HR":
                        scored = 1 + sum(1 for b in bases if b)
                        bases = [None,None,None]
                    elif outcome=="3B":
                        scored = sum(1 for b in bases if b)
                        bases = [None,None,batter.name]
                    elif outcome=="2B":
                        scored = (1 if bases[2] else 0)+(1 if bases[1] else 0)
                        bases = [None,batter.name,bases[0]]
                    elif outcome=="1B":
                        scored += 1 if bases[2] else 0
                        scored += 1 if (bases[1] and random.random()<0.55) else 0
                        bases = [batter.name,bases[0],None]
                    elif outcome=="BB":
                        if all(bases): scored=1
                        if bases[0] and bases[1]: bases[2]=bases[1]
                        if bases[0]: bases[1]=bases[0]
                        bases[0]=batter.name
                    else:
                        outs+=1

                    runs_this_half += scored
                    if half=="top": away_score+=scored
                    else:           home_score+=scored

                    yield emit("plate_appearance", {
                        "inning": inning, "half": half,
                        "batter": batter.name, "team": cfg["away_team" if half=="top" else "home_team"],
                        "pitcher": p.name, "pitch_count": p.pitch_count,
                        "outcome": outcome,
                        "runs_scored": scored,
                        "outs": outs,
                        "bases": bases,
                        "away_score": away_score, "home_score": home_score,
                        "exit_velo": ev_val, "launch_angle": la_val, "distance": dist_val,
                        "dist": dist,
                        "barrel_rate": batter.barrel_rate,
                    })
                    time.sleep(delay)

                linescore["away" if half=="top" else "home"].append(runs_this_half)

                yield emit("half_inning_end", {
                    "inning": inning, "half": half,
                    "runs": runs_this_half,
                    "away_score": away_score, "home_score": home_score,
                    "linescore": linescore,
                })
                time.sleep(delay * 0.3)

            if inning >= 9 and away_score != home_score: break

        winner = cfg["away_team"] if away_score > home_score else cfg["home_team"]
        yield emit("game_over", {
            "away_score": away_score, "home_score": home_score,
            "winner": winner, "linescore": linescore,
        })

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


def advance_runners(outcome, batter_name, bases, outs):
    """Return (runs_scored, new_bases, new_outs)."""
    scored = 0
    bases = list(bases)
    if outcome == "HR":
        scored = 1 + sum(1 for b in bases if b)
        bases = [None, None, None]
    elif outcome == "3B":
        scored = sum(1 for b in bases if b)
        bases = [None, None, batter_name]
    elif outcome == "2B":
        scored = (1 if bases[2] else 0) + (1 if bases[1] else 0)
        bases = [None, batter_name, bases[0]]
    elif outcome == "1B":
        scored += 1 if bases[2] else 0
        scored += 1 if (bases[1] and random.random() < 0.55) else 0
        bases = [batter_name, bases[0], None]
    elif outcome == "BB":
        if bases[0] and bases[1] and bases[2]: scored = 1
        if bases[0] and bases[1]: bases[2] = bases[1]
        if bases[0]: bases[1] = bases[0]
        bases[0] = batter_name
    else:  # K or Out
        outs += 1
    return scored, bases, outs


@app.route("/api/game_events")
def game_events():
    """
    Run the full game sim instantly, return all events as a JSON array.
    The UI fetches this once and replays it client-side — avoids SSE buffering on Railway.
    """
    cfg = GAME_CONFIG
    park_f = PARK_HR_FACTORS.get(cfg["park"], 1.0)
    wm = weather_mult(cfg["temp_f"], cfg["wind_out_mph"])

    away_sp = make_pitcher(cfg["away_starter"])
    home_sp = make_pitcher(cfg["home_starter"])
    away_bp = [make_pitcher(p) for p in cfg["away_bullpen"]]
    home_bp = [make_pitcher(p) for p in cfg["home_bullpen"]]
    home_bp_idx = away_bp_idx = 0

    home_score = away_score = 0
    away_idx = home_idx = 0
    linescore = {"away": [], "home": []}
    events = []

    def active_p(side):
        if side == "home": return away_sp if home_bp_idx==0 else away_bp[home_bp_idx-1]
        else:              return home_sp if away_bp_idx==0 else home_bp[away_bp_idx-1]

    away_lineup = [make_batter(b) for b in cfg["away_lineup"]]
    home_lineup = [make_batter(b) for b in cfg["home_lineup"]]

    events.append({"type": "game_start",
        "away_team": cfg["away_team"], "home_team": cfg["home_team"],
        "away_starter": cfg["away_starter"]["name"],
        "home_starter": cfg["home_starter"]["name"],
        "park": cfg["park"], "park_factor": park_f, "weather_mult": wm,
        "away_lineup": [b.name for b in away_lineup],
        "home_lineup": [b.name for b in home_lineup],
    })

    for inning in range(1, 14):
        for half in ["top", "bottom"]:
            if inning >= 10 and half == "bottom" and home_score > away_score: break
            lineup = away_lineup if half == "top" else home_lineup
            idx_ref = [away_idx if half == "top" else home_idx]
            bp_idx_ref = [home_bp_idx if half == "top" else away_bp_idx]
            pitcher = active_p("home" if half == "top" else "away")
            pitch_count = 0
            outs = 0
            bases = [None, None, None]
            runs_this_half = 0

            events.append({"type": "half_inning_start",
                "inning": inning, "half": half,
                "pitcher": pitcher.name, "pitch_count": pitch_count,
                "away_score": away_score, "home_score": home_score,
            })

            while outs < 3:
                cur_idx = away_idx if half == "top" else home_idx
                batter = lineup[cur_idx % 9]
                if half == "top": away_idx += 1
                else: home_idx += 1

                # Pitching change
                pc = pitch_count
                bp = away_bp if half == "top" else home_bp
                bp_idx = home_bp_idx if half == "top" else away_bp_idx
                if (pc >= pitcher.max_effective_pitches + 10 or
                    (inning >= 6 and pc >= 80 and pitcher.era > 4.8)) and bp_idx < len(bp):
                    if half == "top": home_bp_idx += 1
                    else: away_bp_idx += 1
                    pitcher = active_p("home" if half == "top" else "away")
                    pitch_count = 0
                    events.append({"type": "pitching_change", "pitcher": pitcher.name,
                        "inning": inning, "half": half})

                dist = pa_outcome_dist(batter, pitcher, park_f, wm)
                outcome = roll_outcome(dist)
                pitch_count += 4 if outcome in ("BB","K") else 3

                ev_val = la_val = dist_val = None
                if outcome == "HR":
                    ev_val = round(batter.exit_velo + random.gauss(0,2), 1)
                    la_val = round(random.uniform(25,40), 1)
                    dist_val = int(random.uniform(380,470))
                elif outcome in ("3B","2B","1B","Out"):
                    ev_val = round(batter.exit_velo + random.gauss(-5,4), 1)
                    la_val = round(random.uniform(-5,25), 1)
                    dist_val = int(random.uniform(60,350))

                scored, bases, outs = advance_runners(outcome, batter.name, bases, outs)
                if half == "top": away_score += scored
                else: home_score += scored
                runs_this_half += scored

                events.append({"type": "plate_appearance",
                    "inning": inning, "half": half,
                    "batter": batter.name, "team": cfg["away_team"] if half=="top" else cfg["home_team"],
                    "pitcher": pitcher.name, "pitch_count": pitch_count,
                    "outcome": outcome, "runs_scored": scored,
                    "outs": outs, "bases": bases,
                    "away_score": away_score, "home_score": home_score,
                    "exit_velo": ev_val, "launch_angle": la_val, "distance": dist_val,
                    "dist": dist, "barrel_rate": batter.barrel_rate,
                })

            linescore["away" if half=="top" else "home"].append(runs_this_half)
            events.append({"type": "half_inning_end",
                "inning": inning, "half": half,
                "runs": runs_this_half,
                "away_score": away_score, "home_score": home_score,
                "linescore": linescore,
            })

        if inning >= 9 and away_score != home_score: break

    winner = cfg["away_team"] if away_score > home_score else cfg["home_team"]
    events.append({"type": "game_over",
        "away_score": away_score, "home_score": home_score,
        "winner": winner, "linescore": linescore,
    })

    return jsonify(events)


@app.route("/api/status")
def status():
    return jsonify({"status": "ok", "game": f"{GAME_CONFIG['away_team']} @ {GAME_CONFIG['home_team']}", "date": GAME_CONFIG["date"]})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    print(f"\n🔴 MLB Sim API running at http://0.0.0.0:{port}")
    print("   Endpoints:")
    print("   GET  /api/game       — game config + lineups")
    print("   GET  /api/pa_dist    — PA outcome distribution")
    print("   POST /api/simulate   — Monte Carlo aggregates")
    print("   GET  /api/live_game  — SSE stream of live game\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
