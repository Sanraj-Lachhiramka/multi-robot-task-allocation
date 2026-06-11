"""
Distributed Task Allocation for Multi-Robot Systems under Communication Constraints
====================================================================================
Novel research direction: decentralized multi-robot task allocation where robots
must coordinate under intermittent, range-limited communication.

Three methods compared:
  1. Greedy Nearest-Task (baseline 1) -- each robot picks closest unassigned task
  2. Auction-based (baseline 2)       -- robots bid; highest bidder wins task
  3. Consensus-based Voronoi (proposed) -- robots partition space via Voronoi,
     then allocate within their region; handles communication loss gracefully

Author: Sanraj Lachhiramka
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.colors import ListedColormap
import time, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
AREA       = 100.0        # square environment side length (m)
N_ROBOTS   = 5
N_TASKS    = 20
COMM_RANGE = 40.0         # communication radius (m)
N_TRIALS   = 40           # Monte Carlo trials for statistics
COLORS     = ['#e41a1c','#377eb8','#4daf4a','#ff7f00','#984ea3']

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def total_travel(robots, assignment):
    """Sum of travel distances for all robots across their assigned tasks."""
    total = 0.0
    for r_idx, task_list in assignment.items():
        pos = robots[r_idx].copy()
        for t in task_list:
            total += dist(pos, t)
            pos = t
    return total

def makespan(robots, assignment):
    """Time for the last robot to finish (assumes unit speed)."""
    times = []
    for r_idx, task_list in assignment.items():
        d = 0.0
        pos = robots[r_idx].copy()
        for t in task_list:
            d += dist(pos, t)
            pos = t
        times.append(d)
    return max(times) if times else 0.0

def comm_graph(robots, comm_range):
    """Returns adjacency dict for robots within comm_range."""
    n = len(robots)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            if dist(robots[i], robots[j]) <= comm_range:
                adj[i].append(j)
                adj[j].append(i)
    return adj

# ─────────────────────────────────────────────────────────────
# METHOD 1: GREEDY NEAREST-TASK
# ─────────────────────────────────────────────────────────────

def greedy_nearest(robots, tasks):
    """
    Each robot, in round-robin turns, picks the nearest unassigned task.
    Simple, deterministic, ignores communication.
    """
    assignment = {i: [] for i in range(len(robots))}
    positions  = [r.copy() for r in robots]
    remaining  = list(range(len(tasks)))

    while remaining:
        for r_idx in range(len(robots)):
            if not remaining:
                break
            # find nearest unassigned task
            nearest = min(remaining, key=lambda t: dist(positions[r_idx], tasks[t]))
            assignment[r_idx].append(tasks[nearest])
            positions[r_idx] = tasks[nearest].copy()
            remaining.remove(nearest)

    return assignment

# ─────────────────────────────────────────────────────────────
# METHOD 2: AUCTION-BASED ALLOCATION
# ─────────────────────────────────────────────────────────────

def auction_based(robots, tasks, comm_range):
    """
    Sequential single-item auctions.
    Each robot broadcasts a bid (inverse distance to task).
    Only robots reachable via the communication graph can bid.
    Highest bidder wins. Communication-limited version.
    """
    assignment   = {i: [] for i in range(len(robots))}
    positions    = [r.copy() for r in robots]
    remaining    = list(range(len(tasks)))
    robot_loads  = [0.0] * len(robots)   # accumulated travel cost

    while remaining:
        adj = comm_graph(positions, comm_range)

        for t_idx in remaining[:]:
            # Bids: robots bid based on marginal cost
            bids = {}
            for r_idx in range(len(robots)):
                # Only bid if robot can communicate with at least one neighbour
                # (simulates needing network connectivity to participate)
                if len(adj[r_idx]) > 0 or len(robots) == 1:
                    marginal = dist(positions[r_idx], tasks[t_idx])
                    bids[r_idx] = -marginal  # higher bid = closer

            if not bids:
                # No robot can communicate -- assign to globally closest as fallback
                closest = min(range(len(robots)),
                              key=lambda r: dist(positions[r], tasks[t_idx]))
                bids[closest] = 0

            winner = max(bids, key=bids.get)
            assignment[winner].append(tasks[t_idx])
            robot_loads[winner] += dist(positions[winner], tasks[t_idx])
            positions[winner] = tasks[t_idx].copy()
            remaining.remove(t_idx)
            break   # one auction per outer loop to allow position updates

    return assignment

# ─────────────────────────────────────────────────────────────
# METHOD 3: CONSENSUS-BASED VORONOI PARTITIONING (PROPOSED)
# ─────────────────────────────────────────────────────────────

def voronoi_consensus(robots, tasks, comm_range, area=AREA):
    """
    Step 1 – Partition: assign each task to nearest robot (Voronoi ownership).
    Step 2 – Local optimise: within each robot's partition, solve nearest-
             neighbour tour (greedy TSP approximation).
    Step 3 – Rebalance: robots that share a communication link and have
             imbalanced loads exchange boundary tasks to equalise makespan.
    This is communication-aware: only connected neighbours rebalance.
    """
    n = len(robots)
    assignment = {i: [] for i in range(n)}

    # Step 1: Voronoi assignment
    for t in tasks:
        owner = min(range(n), key=lambda r: dist(robots[r], t))
        assignment[owner].append(t)

    # Step 2: within each partition, greedy nearest-neighbour ordering
    ordered = {}
    for r_idx in range(n):
        pos = robots[r_idx].copy()
        remaining = list(range(len(assignment[r_idx])))
        task_arr  = assignment[r_idx]
        tour = []
        while remaining:
            nearest_idx = min(remaining, key=lambda i: dist(pos, task_arr[i]))
            tour.append(task_arr[nearest_idx])
            pos = task_arr[nearest_idx].copy()
            remaining.remove(nearest_idx)
        ordered[r_idx] = tour

    # Step 3: rebalance via communication graph (one pass)
    adj = comm_graph(robots, comm_range)
    robot_costs = {r: sum(dist(robots[r] if i==0 else ordered[r][i-1],
                               ordered[r][i])
                          for i in range(len(ordered[r])))
                   for r in range(n)}

    for r_idx in range(n):
        for neighbour in adj[r_idx]:
            # if r_idx is heavily loaded vs neighbour, transfer one boundary task
            if robot_costs[r_idx] > robot_costs[neighbour] * 1.3 and ordered[r_idx]:
                # transfer last task (boundary of tour) -- use index to avoid array comparison
                transfer_idx = len(ordered[r_idx]) - 1
                task_to_transfer = ordered[r_idx][transfer_idx]
                ordered[r_idx] = ordered[r_idx][:transfer_idx]
                ordered[neighbour].append(task_to_transfer)
                # update costs roughly
                robot_costs[r_idx]    *= 0.85
                robot_costs[neighbour] *= 1.15

    return ordered

# ─────────────────────────────────────────────────────────────
# MONTE CARLO EVALUATION
# ─────────────────────────────────────────────────────────────

def run_trial(seed):
    rng   = np.random.default_rng(seed)
    robots = [rng.uniform(0, AREA, 2) for _ in range(N_ROBOTS)]
    tasks  = [rng.uniform(0, AREA, 2) for _ in range(N_TASKS)]

    a1 = greedy_nearest(robots, tasks)
    a2 = auction_based(robots, tasks, COMM_RANGE)
    a3 = voronoi_consensus(robots, tasks, COMM_RANGE)

    return {
        'greedy':   (total_travel(robots, a1), makespan(robots, a1)),
        'auction':  (total_travel(robots, a2), makespan(robots, a2)),
        'voronoi':  (total_travel(robots, a3), makespan(robots, a3)),
        'robots': robots, 'tasks': tasks,
        'a1': a1, 'a2': a2, 'a3': a3,
    }

print("Running Monte Carlo trials...")
results = [run_trial(s) for s in range(N_TRIALS)]

greedy_travel  = [r['greedy'][0]  for r in results]
auction_travel = [r['auction'][0] for r in results]
voronoi_travel = [r['voronoi'][0] for r in results]

greedy_make  = [r['greedy'][1]  for r in results]
auction_make = [r['auction'][1] for r in results]
voronoi_make = [r['voronoi'][1] for r in results]

print(f"Greedy   — travel: {np.mean(greedy_travel):.1f}±{np.std(greedy_travel):.1f}  makespan: {np.mean(greedy_make):.1f}±{np.std(greedy_make):.1f}")
print(f"Auction  — travel: {np.mean(auction_travel):.1f}±{np.std(auction_travel):.1f}  makespan: {np.mean(auction_make):.1f}±{np.std(auction_make):.1f}")
print(f"Voronoi  — travel: {np.mean(voronoi_travel):.1f}±{np.std(voronoi_travel):.1f}  makespan: {np.mean(voronoi_make):.1f}±{np.std(voronoi_make):.1f}")

# ─────────────────────────────────────────────────────────────
# FIGURE 1: Example single-scenario visualisation (3 subplots)
# ─────────────────────────────────────────────────────────────
sample = results[0]
robots = sample['robots']
tasks  = sample['tasks']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
method_names = ['Greedy Nearest-Task', 'Auction-Based', 'Voronoi-Consensus (Proposed)']
assignments  = [sample['a1'], sample['a2'], sample['a3']]
metrics      = [sample['greedy'], sample['auction'], sample['voronoi']]

for ax, name, asgn, met in zip(axes, method_names, assignments, metrics):
    ax.set_xlim(0, AREA); ax.set_ylim(0, AREA)
    ax.set_facecolor('#f8f9fa')
    ax.set_title(name, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')

    # draw communication circles (light)
    for r_idx, r in enumerate(robots):
        circle = plt.Circle(r, COMM_RANGE, color=COLORS[r_idx],
                            alpha=0.07, linewidth=0)
        ax.add_patch(circle)

    # draw task paths per robot
    for r_idx, task_list in asgn.items():
        if not task_list:
            continue
        path = [robots[r_idx]] + task_list
        xs = [p[0] for p in path]; ys = [p[1] for p in path]
        ax.plot(xs, ys, '-', color=COLORS[r_idx], alpha=0.6, linewidth=1.5)
        for i, (x, y) in enumerate(zip(xs[1:], ys[1:])):
            ax.annotate('', xy=(x, y), xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle='->', color=COLORS[r_idx],
                                        lw=1.2, alpha=0.7))

    # tasks
    for t in tasks:
        ax.plot(*t, 's', color='#333333', markersize=5, zorder=3)

    # robots
    for r_idx, r in enumerate(robots):
        ax.plot(*r, 'o', color=COLORS[r_idx], markersize=10,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax.text(r[0]+1.5, r[1]+1.5, f'R{r_idx+1}', fontsize=8,
                color=COLORS[r_idx], fontweight='bold')

    ax.text(0.02, 0.97, f'Travel: {met[0]:.0f} m\nMakespan: {met[1]:.0f} m',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.grid(True, alpha=0.3, linestyle='--')

# robot legend
handles = [mpatches.Patch(color=COLORS[i], label=f'Robot {i+1}') for i in range(N_ROBOTS)]
handles.append(plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='#333',
                           markersize=7, label='Task'))
fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=9,
           bbox_to_anchor=(0.5, -0.04))
plt.suptitle('Multi-Robot Task Allocation: Single Scenario Comparison', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('ms_sim/fig1_scenario.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_scenario.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 2: Monte Carlo boxplots (travel + makespan)
# ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

bp_colors = ['#e74c3c', '#3498db', '#2ecc71']
labels    = ['Greedy', 'Auction', 'Voronoi\n(Proposed)']

# Travel distance
bp1 = ax1.boxplot([greedy_travel, auction_travel, voronoi_travel],
                   labels=labels, patch_artist=True, notch=False,
                   medianprops=dict(color='black', linewidth=2))
for patch, c in zip(bp1['boxes'], bp_colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax1.set_title('Total Travel Distance', fontweight='bold')
ax1.set_ylabel('Total distance (m)')
ax1.grid(True, axis='y', alpha=0.4, linestyle='--')
ax1.yaxis.set_major_locator(plt.MaxNLocator(6))

# Makespan
bp2 = ax2.boxplot([greedy_make, auction_make, voronoi_make],
                   labels=labels, patch_artist=True, notch=False,
                   medianprops=dict(color='black', linewidth=2))
for patch, c in zip(bp2['boxes'], bp_colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax2.set_title('Makespan (Completion Time)', fontweight='bold')
ax2.set_ylabel('Makespan (m / unit speed)')
ax2.grid(True, axis='y', alpha=0.4, linestyle='--')
ax2.yaxis.set_major_locator(plt.MaxNLocator(6))

plt.suptitle(f'Performance over {N_TRIALS} Random Trials\n'
             f'({N_ROBOTS} robots, {N_TASKS} tasks, comm range = {COMM_RANGE} m)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('ms_sim/fig2_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_boxplots.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 3: Effect of communication range on makespan
# ─────────────────────────────────────────────────────────────
comm_ranges = [10, 20, 30, 40, 50, 70, 100]
greedy_ms_cr, auction_ms_cr, voronoi_ms_cr = [], [], []

for cr in comm_ranges:
    g_vals, a_vals, v_vals = [], [], []
    for seed in range(20):
        rng    = np.random.default_rng(seed + 100)
        robots = [rng.uniform(0, AREA, 2) for _ in range(N_ROBOTS)]
        tasks  = [rng.uniform(0, AREA, 2) for _ in range(N_TASKS)]
        a1 = greedy_nearest(robots, tasks)
        a2 = auction_based(robots, tasks, cr)
        a3 = voronoi_consensus(robots, tasks, cr)
        g_vals.append(makespan(robots, a1))
        a_vals.append(makespan(robots, a2))
        v_vals.append(makespan(robots, a3))
    greedy_ms_cr.append(np.mean(g_vals))
    auction_ms_cr.append(np.mean(a_vals))
    voronoi_ms_cr.append(np.mean(v_vals))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(comm_ranges, greedy_ms_cr,  'o-', color='#e74c3c', label='Greedy',            lw=2)
ax.plot(comm_ranges, auction_ms_cr, 's-', color='#3498db', label='Auction',           lw=2)
ax.plot(comm_ranges, voronoi_ms_cr, '^-', color='#2ecc71', label='Voronoi (Proposed)', lw=2.5)
ax.set_xlabel('Communication Range (m)', fontsize=11)
ax.set_ylabel('Mean Makespan (m)', fontsize=11)
ax.set_title('Effect of Communication Range on Makespan\n'
             f'({N_ROBOTS} robots, {N_TASKS} tasks, avg over 20 trials)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4, linestyle='--')
ax.fill_between(comm_ranges, voronoi_ms_cr, greedy_ms_cr,
                alpha=0.07, color='#2ecc71', label='_nolegend_')
plt.tight_layout()
plt.savefig('ms_sim/fig3_comm_range.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig3_comm_range.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 4: Load balancing (per-robot makespan variance)
# ─────────────────────────────────────────────────────────────
def per_robot_makespan(robots, assignment):
    ms = []
    for r_idx in range(len(robots)):
        task_list = assignment[r_idx]
        d = 0.0; pos = robots[r_idx].copy()
        for t in task_list:
            d += dist(pos, t); pos = t
        ms.append(d)
    return ms

g_vars, a_vars, v_vars = [], [], []
for r in results:
    g_vars.append(np.std(per_robot_makespan(r['robots'], r['a1'])))
    a_vars.append(np.std(per_robot_makespan(r['robots'], r['a2'])))
    v_vars.append(np.std(per_robot_makespan(r['robots'], r['a3'])))

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(3)
means = [np.mean(g_vars), np.mean(a_vars), np.mean(v_vars)]
stds  = [np.std(g_vars),  np.std(a_vars),  np.std(v_vars)]
bars  = ax.bar(x, means, yerr=stds, color=bp_colors, alpha=0.75,
               capsize=6, edgecolor='grey', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Std Dev of per-robot makespan (m)', fontsize=10)
ax.set_title('Load Balance Quality (lower = more balanced)\n'
             f'Mean ± Std over {N_TRIALS} trials', fontsize=11, fontweight='bold')
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{m:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.grid(True, axis='y', alpha=0.4, linestyle='--')
plt.tight_layout()
plt.savefig('ms_sim/fig4_load_balance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_load_balance.png")

# ─────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY (mean ± std over 40 trials)")
print("="*60)
print(f"{'Method':<20} {'Travel (m)':>15} {'Makespan (m)':>15} {'Load Std':>12}")
print("-"*60)
for name, tv, ms, lv in [
    ('Greedy',          greedy_travel,  greedy_make,  g_vars),
    ('Auction',         auction_travel, auction_make, a_vars),
    ('Voronoi (Prop.)', voronoi_travel, voronoi_make, v_vars),
]:
    print(f"{name:<20} {np.mean(tv):>10.1f}±{np.std(tv):<4.1f} "
          f"{np.mean(ms):>10.1f}±{np.std(ms):<4.1f} "
          f"{np.mean(lv):>8.1f}±{np.std(lv):.1f}")
print("="*60)
