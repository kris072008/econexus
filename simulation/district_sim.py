"""
EcoNexus district simulation (rebuilt, v2).

The original SimPy model was lost; this is a new model built from the description
in the paper, with a rate-based sorting station and the sorting rates measured
offline on TrashNet (paper, Section IV).

What it models
  * 12 communal hubs, each with one 660 L dry-recyclables bin with a fill sensor.
  * Hourly deposits (06:00-23:00) with per-hub daily variation and district-wide
    surge days (markets, public holidays).
  * One collection truck based at the sorting station. Each dispatch is routed with
    OR-Tools (capacitated VRP) over the hubs to be collected.
  * Three collection strategies on the SAME generated waste (common random numbers):
      fixed_3x    - every hub, Mon/Wed/Fri at 08:00 (NEA minimum for 660 L bins)
      fixed_daily - every hub, daily at 08:00
      fill_naive  - daily 08:00 check; collect only hubs whose sensor reads >= fill_threshold
      fill_based  - same trigger, but a triggered trip also collects hubs >= co_collect_threshold
  * A rate-based sorting station (items/hour) fed by the truck; sorting outcomes per
    item use the measured TrashNet rates for three policies (ungated, gated k=3,
    gated k=9), applied to the same processed items.

Every parameter is in PARAMS with its source or marked ASSUMED.

Usage
  pip install simpy ortools numpy pandas
  python district_sim.py                 # 10 seeds, writes results/ next to this file
  python district_sim.py --seeds 3 --quick
"""
import argparse, json, math, os, time
from functools import lru_cache
import numpy as np, pandas as pd, simpy
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

PARAMS = {
    # --- district ---
    'n_hubs': 12,                      # paper, Section III-E
    'households_per_hub': 42,          # paper, Section III-E
    'persons_per_household': 3.06,     # SingStat, resident households, 2025
    'waste_kg_per_person_day': 0.83,   # NEA, domestic waste generated per capita per day, 2025
    'share_to_recycling_bin': 0.25,    # ASSUMED: share of domestic waste (by mass) placed in the dry-recyclables bin
    'bin_capacity_L': 660,             # NEA National Recycling Programme: 660 L blue bin per HDB block
    'density_kg_per_L': 0.155,         # US EPA (2016) volume-to-weight: commingled recyclables 262 lb/yd3 = 155 kg/m3
    'item_mass_kg': 0.040,             # ASSUMED: mean mass of one recyclable item
    'deposit_hours': (6, 23),          # ASSUMED: deposits between 06:00 and 23:00
    'hub_daily_cv': 0.25,              # ASSUMED: day-to-day variation per hub (lognormal sigma)
    'surge_prob_per_day': 0.05,        # ASSUMED: probability that a day is a surge day
    'surge_multiplier': 1.8,           # ASSUMED: generation multiplier on surge days
    'district_side_km': 1.5,           # ASSUMED: hubs spread over a 1.5 km x 1.5 km district
    'depot_offset_km': 2.0,            # ASSUMED: station/depot 2 km from district centre
    'circuity_factor': 1.3,            # ASSUMED: road distance / straight-line distance
    'layout_seed': 42,                 # fixed hub layout for all runs
    # --- truck ---
    'truck_capacity_L': 16000,         # Sousa et al. (2023): 16 m3 collection trucks (no compaction assumed)
    'truck_speed_kmh': 20.0,           # ASSUMED: urban collection speed
    'service_min_per_hub': 5.0,        # ASSUMED: time to empty one bin
    'fuel_L_per_km': 1.004,            # Sousa et al. (2023): mean 100.40 L/100 km, waste collection trucks
    'co2_kg_per_L_diesel': 2.68,       # IPCC 2006 Vol.2 Ch.3: 74,100 kg CO2/TJ x 43.0 TJ/Gg NCV, density 0.84 kg/L (ASSUMED)
    # --- collection strategies ---
    'fixed_3x_days': (0, 2, 4),        # Mon, Wed, Fri - NEA: 660 L bins "collected at least thrice a week"
    'dispatch_hour': 8,                # ASSUMED
    'fill_threshold': 0.60,            # ASSUMED: a hub at or above this fill triggers a dispatch
    'co_collect_threshold': 0.30,      # ASSUMED: on a triggered dispatch, also collect hubs at or above this fill
    # --- sorting station ---
    'station_hours': (6, 22),          # ASSUMED: 16 operating hours per day
    'station_items_per_hour': 800,     # design parameter, sized for utilisation below 1 (see capacity sweep)
    # --- sorting outcomes (paper, Section IV, Table VI; TrashNet test sets, mean of 3 seeds) ---
    'policies': {                      # share admitted to recycling, contamination of admitted
        'ungated':   {'admitted': 0.5886, 'contamination': 0.0491},
        'gated_k3':  {'admitted': 0.5342, 'contamination': 0.0181},
        'gated_k9':  {'admitted': 0.4904, 'contamination': 0.0085},
    },
    # --- run control ---
    'days': 120, 'warmup_days': 30,
}
STRATEGIES = ['fixed_3x', 'fixed_daily', 'fill_naive', 'fill_based']


def layout(P):
    rng = np.random.default_rng(P['layout_seed'])
    s = P['district_side_km']
    hubs = rng.uniform(-s / 2, s / 2, size=(P['n_hubs'], 2))
    depot = np.array([[P['depot_offset_km'], 0.0]])
    pts = np.vstack([depot, hubs])                                   # node 0 = depot
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2) * P['circuity_factor']
    return pts, d                                                     # km


def generate_deposits(P, seed):
    """Litres deposited per hub per hour, shape (days*24, n_hubs). Shared by all strategies."""
    rng = np.random.default_rng(seed)
    mean_L_day = (P['households_per_hub'] * P['persons_per_household'] * P['waste_kg_per_person_day']
                  * P['share_to_recycling_bin'] / P['density_kg_per_L'])
    h0, h1 = P['deposit_hours']; nh = h1 - h0
    days, n = P['days'], P['n_hubs']
    sig = P['hub_daily_cv']
    daily = mean_L_day * rng.lognormal(-sig ** 2 / 2, sig, size=(days, n))
    surge = rng.random(days) < P['surge_prob_per_day']
    daily[surge] *= P['surge_multiplier']
    dep = np.zeros((days * 24, n))
    for d in range(days):
        # spread each hub's daily volume over deposit hours as Poisson-sized bags (~5 L each, ASSUMED)
        bags = rng.poisson(daily[d] / 5.0)
        for j in range(n):
            hrs = rng.integers(h0, h1, size=bags[j])
            np.add.at(dep[:, j], d * 24 + hrs, 5.0)
    return dep, surge, mean_L_day


class Router:
    def __init__(self, dist_km, cap_L):
        self.D = np.rint(dist_km * 1000).astype(int)               # metres
        self.cap = int(cap_L)

    @lru_cache(maxsize=8192)
    def route(self, nodes, loads):
        """nodes: tuple of hub node ids (1..n); loads: tuple of litres. Returns (ordered stops per trip, km)."""
        idx = [0] + list(nodes)
        dem = [0] + [int(math.ceil(l)) for l in loads]
        nv = max(1, math.ceil(sum(dem) / self.cap) + 1)
        man = pywrapcp.RoutingIndexManager(len(idx), nv, 0)
        r = pywrapcp.RoutingModel(man)
        cb = r.RegisterTransitCallback(lambda a, b: int(self.D[idx[man.IndexToNode(a)], idx[man.IndexToNode(b)]]))
        r.SetArcCostEvaluatorOfAllVehicles(cb)
        dcb = r.RegisterUnaryTransitCallback(lambda a: dem[man.IndexToNode(a)])
        r.AddDimensionWithVehicleCapacity(dcb, 0, [self.cap] * nv, True, 'cap')
        prm = pywrapcp.DefaultRoutingSearchParameters()
        prm.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        prm.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        prm.time_limit.FromMilliseconds(100)
        sol = r.SolveWithParameters(prm)
        trips, total = [], 0
        for v in range(nv):
            i = r.Start(v); trip = []
            while not r.IsEnd(i):
                nxt = sol.Value(r.NextVar(i)); total += r.GetArcCostForVehicle(i, nxt, v)
                if man.IndexToNode(i) != 0: trip.append(idx[man.IndexToNode(i)])
                i = nxt
            if trip: trips.append(trip)
        return trips, total / 1000.0


def run(P, seed, strategy, router, D_km, deposits, station_rate=None):
    env = simpy.Environment()
    n = P['n_hubs']; cap = P['bin_capacity_L']
    rate = station_rate or P['station_items_per_hour']
    items_per_L = P['density_kg_per_L'] / P['item_mass_kg']
    rng = np.random.default_rng(10_000 + seed)
    fill = np.zeros(n)
    st = {'km': np.zeros(P['days']), 'dispatches': np.zeros(P['days']), 'visits': np.zeros(P['days']),
          'overflow_hubdays': np.zeros(P['days']), 'overflow_L': np.zeros(P['days']),
          'collected_L': np.zeros(P['days']), 'fill_at_collection': [], 'processed': np.zeros(P['days']),
          'backlog_eod': np.zeros(P['days']), 'deposited_L': np.zeros(P['days'])}
    outcomes = {k: {'recovered': np.zeros(P['days']), 'contaminants': np.zeros(P['days'])} for k in P['policies']}
    queue = [0.0]
    overflowed_today = np.zeros(n, bool)

    def generation():
        for h in range(P['days'] * 24):
            d = h // 24
            if h % 24 == 0: overflowed_today[:] = False
            add = deposits[h]
            st['deposited_L'][d] += add.sum()
            room = cap - fill
            over = np.maximum(add - room, 0)
            fill[:] = np.minimum(fill + add, cap)
            newly = (over > 0) & ~overflowed_today
            st['overflow_hubdays'][d] += newly.sum(); overflowed_today[:] |= over > 0
            st['overflow_L'][d] += over.sum()
            yield env.timeout(1)

    def truck(nodes):
        d = int(env.now // 24)
        loads = tuple(float(fill[j - 1]) for j in nodes)
        # loads are rounded to 1,000 L: only used for the capacity check, and rounding improves cache hits
        trips, km = router.route(tuple(nodes), tuple(round(l, -3) for l in loads))
        st['km'][d] += km; st['dispatches'][d] += 1
        speed = P['truck_speed_kmh']; pos = 0
        for trip in trips:
            for node in trip:
                yield env.timeout(D_km[pos, node] / speed + P['service_min_per_hub'] / 60)
                vol = fill[node - 1]
                st['fill_at_collection'].append(vol / cap)
                st['collected_L'][d] += vol; st['visits'][d] += 1
                fill[node - 1] = 0.0
                carried.append(vol); pos = node
            yield env.timeout(D_km[pos, 0] / speed); pos = 0
            queue[0] += sum(carried) * items_per_L; carried.clear()

    carried = []

    def dispatcher():
        yield env.timeout(P['dispatch_hour'])
        while env.now < P['days'] * 24:
            d = int(env.now // 24); dow = d % 7
            if strategy == 'fixed_daily' or (strategy == 'fixed_3x' and dow in P['fixed_3x_days']):
                nodes = list(range(1, n + 1))
            elif strategy in ('fill_naive', 'fill_based'):
                f = fill / cap
                if (f >= P['fill_threshold']).any():
                    lo = P['fill_threshold'] if strategy == 'fill_naive' else P['co_collect_threshold']
                    nodes = [j + 1 for j in range(n) if f[j] >= lo]
                else:
                    nodes = []
            else:
                nodes = []
            if nodes: env.process(truck(nodes))
            yield env.timeout(24)

    def station():
        a, b = P['station_hours']
        for h in range(P['days'] * 24):
            d = h // 24
            if a <= h % 24 < b:
                done = min(queue[0], rate); queue[0] -= done
                st['processed'][d] += done
                k = int(round(done))
                for name, pol in P['policies'].items():
                    adm = rng.binomial(k, pol['admitted'])
                    cont = rng.binomial(adm, pol['contamination'])
                    outcomes[name]['recovered'][d] += adm - cont
                    outcomes[name]['contaminants'][d] += cont
            if h % 24 == 23: st['backlog_eod'][d] = queue[0]
            yield env.timeout(1)

    env.process(generation()); env.process(dispatcher()); env.process(station())
    env.run(until=P['days'] * 24)

    w = slice(P['warmup_days'], P['days']); nd = P['days'] - P['warmup_days']
    per30 = 30.0 / nd
    km = st['km'][w].sum()
    dep_items = st['deposited_L'][w].sum() * items_per_L
    half = P['warmup_days'] + nd // 2
    res = {'seed': seed, 'strategy': strategy, 'station_items_per_hour': rate,
           'distance_km_per30d': km * per30,
           'co2_kg_per30d': km * P['fuel_L_per_km'] * P['co2_kg_per_L_diesel'] * per30,
           'dispatches_per30d': st['dispatches'][w].sum() * per30,
           'hub_visits_per30d': st['visits'][w].sum() * per30,
           'overflow_hubdays_per30d': st['overflow_hubdays'][w].sum() * per30,
           'overflow_share_%': 100 * st['overflow_L'][w].sum() / st['deposited_L'][w].sum(),
           'mean_fill_at_collection_%': 100 * np.mean(st['fill_at_collection']) if st['fill_at_collection'] else np.nan,
           'items_processed_per30d': st['processed'][w].sum() * per30,
           'station_utilisation_%': 100 * st['processed'][w].sum() / (rate * (P['station_hours'][1] - P['station_hours'][0]) * nd),
           'mean_backlog_items': st['backlog_eod'][w].mean(),
           'backlog_first_half': st['backlog_eod'][P['warmup_days']:half].mean(),
           'backlog_second_half': st['backlog_eod'][half:P['days']].mean(),
           'backlog_end_items': st['backlog_eod'][-1],
           'deposited_items_per30d': dep_items * per30}
    for name in P['policies']:
        rec = outcomes[name]['recovered'][w].sum(); con = outcomes[name]['contaminants'][w].sum()
        res[f'{name}_recovered_kg_per30d'] = rec * P['item_mass_kg'] * per30
        res[f'{name}_contamination_%'] = 100 * con / max(rec + con, 1)
    return res


def summarise(df, keys, cols):
    g = df.groupby(keys)[cols].agg(['mean', 'std'])
    out = pd.DataFrame(index=g.index)
    for c in cols:
        out[c] = [f'{m:,.1f} ± {s:,.1f}' for m, s in zip(g[(c, 'mean')], g[(c, 'std')])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--quick', action='store_true', help='skip the capacity sweep')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
    a = ap.parse_args()
    P = PARAMS; os.makedirs(a.out, exist_ok=True)
    pts, D = layout(P); router = Router(D, P['truck_capacity_L'])
    rows, t0 = [], time.time()
    for s in range(a.seeds):
        dep, surge, mean_L = generate_deposits(P, s)
        for strat in STRATEGIES:
            rows.append(run(P, s, strat, router, D, dep))
        print(f'seed {s} done ({time.time() - t0:.0f}s)')
    df = pd.DataFrame(rows); df.to_csv(f'{a.out}/runs.csv', index=False)

    coll_cols = ['distance_km_per30d', 'co2_kg_per30d', 'dispatches_per30d', 'hub_visits_per30d',
                 'overflow_hubdays_per30d', 'overflow_share_%', 'mean_fill_at_collection_%',
                 'station_utilisation_%', 'mean_backlog_items', 'backlog_end_items']
    t_coll = summarise(df, 'strategy', coll_cols).loc[STRATEGIES]
    t_coll.to_csv(f'{a.out}/table_collection.csv')
    pol_rows = []
    for strat in STRATEGIES:
        for name in P['policies']:
            sub = df[df.strategy == strat]
            r = sub[f'{name}_recovered_kg_per30d']; c = sub[f'{name}_contamination_%']
            pol_rows.append({'strategy': strat, 'policy': name,
                             'recovered_kg_per30d': f'{r.mean():,.1f} ± {r.std():,.1f}',
                             'contamination_%': f'{c.mean():.2f} ± {c.std():.2f}'})
    t_pol = pd.DataFrame(pol_rows); t_pol.to_csv(f'{a.out}/table_sorting_policies.csv', index=False)

    demand_items_per_h = df['deposited_items_per30d'].mean() / 30 / (P['station_hours'][1] - P['station_hours'][0])
    info = {'mean_L_per_hub_day': mean_L, 'demand_items_per_station_hour': demand_items_per_h,
            'surge_days_seed0': int(surge.sum()) if a.seeds == 1 else None,
            'hub_layout_km': pts.round(3).tolist(), 'params': {k: v for k, v in P.items()}}
    if not a.quick:
        sweep = []
        for rate in [450, 550, 650, 800, 1000]:
            for s in range(a.seeds):
                dep, _, _ = generate_deposits(P, s)
                sweep.append(run(P, s, 'fill_based', router, D, dep, station_rate=rate))
        sw = pd.DataFrame(sweep); sw.to_csv(f'{a.out}/runs_capacity_sweep.csv', index=False)
        t_sw = summarise(sw, 'station_items_per_hour', ['station_utilisation_%', 'mean_backlog_items',
                                                         'backlog_first_half', 'backlog_second_half', 'backlog_end_items'])
        t_sw.insert(0, 'offered_load_rho', [round(demand_items_per_h / r, 2) for r in t_sw.index])
        t_sw.to_csv(f'{a.out}/table_capacity_sweep.csv')
        print(t_sw.to_string())
    json.dump(info, open(f'{a.out}/run_info.json', 'w'), indent=2, default=str)
    print(t_coll.T.to_string()); print(t_pol.to_string())
    print(json.dumps({k: info[k] for k in ('mean_L_per_hub_day', 'demand_items_per_station_hour')}, indent=1))


if __name__ == '__main__':
    main()
