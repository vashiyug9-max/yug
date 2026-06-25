MAP_FILE = "~/robo_project_ws/src/resources/curve_map1.npy"
START    = (15, 5)
GOAL     = (45, 45)
MODE     = "smooth"    # "discrete" | "smooth" | "viz"

# ================================================================
# IMPORTS
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from collections import deque
from scipy.ndimage import distance_transform_edt

from astar import Astar, Cell, PosePixels

try:
    import rclpy
    from rclpy.node        import Node
    from geometry_msgs.msg import TwistStamped, TransformStamped
    from nav_msgs.msg      import Path, OccupancyGrid
    from geometry_msgs.msg import PoseStamped
    from tf2_ros           import StaticTransformBroadcaster, TransformBroadcaster
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

# ================================================================
# DISCRETE MODE CALIBRATION
# ================================================================
CELL_SIZE_M      = 0.3
STEP_SIZE_M      = 0.1
STEPS_PER_CELL   = int(CELL_SIZE_M / STEP_SIZE_M)
DEGREES_PER_TURN = 15           # max degrees per turn (belt robot)
TURNS_FOR_90     = int(round(90 / DEGREES_PER_TURN))

# ================================================================
# SMOOTH MODE CALIBRATION
# ================================================================
MAX_TURN_DEG = 12.0   # max degrees per sim step
LOOKAHEAD    = 7      # cells to look ahead

# ================================================================
# ROBOT HARDWARE
# ================================================================
LINEAR_VEL   = 0.08   # m/s
ANGULAR_MAX  = 0.35   # rad/s
CMD_TOPIC    = "/diff_cont/cmd_vel"

# ================================================================
# OUTPUT FILES
# ================================================================
OUTPUT_PIPELINE  = "~/robo_project_ws/src/resources/pipeline_result.png"
OUTPUT_DISCRETE  = "~/robo_project_ws/src/resources/discrete_result.png"
OUTPUT_SMOOTH    = "~/robo_project_ws/src/resources/smooth_result.png"
OUTPUT_VIZ       = "~/robo_project_ws/src/resources/movement_cost_viz.png"
CSV_DISCRETE     = "~/robo_project_ws/src/resources/discrete_sequence.csv"
CSV_SMOOTH       = "~/robo_project_ws/src/resources/smooth_sequence.csv"


# ================================================================
# ── SHARED UTILITIES ────────────────────────────────────────────
# ================================================================

def load_map(fname):
    raw  = np.load(fname)
    raw  = np.array(raw, dtype=float)
    if raw.max() > 1:
        raw = raw / raw.max()
    grid = np.round(raw).astype(int)
    print(f"Map loaded     : {grid.shape[0]} x {grid.shape[1]}")
    print(f"Free cells     : {np.sum(grid==1)}")
    print(f"Obstacles      : {np.sum(grid==0)}")
    return grid


def validate(grid, pos, name):
    H, W = grid.shape
    r, c = pos
    if not (0 <= r < H and 0 <= c < W):
        raise ValueError(
            f"{name} {pos} is outside map bounds! "
            f"Valid range: row 0-{H-1}, col 0-{W-1}"
        )
    if grid[r, c] == 0:
        free    = np.argwhere(grid == 1)
        dists   = np.sqrt((free[:,0]-r)**2 + (free[:,1]-c)**2)
        snapped = tuple(free[np.argmin(dists)])
        print(f"WARNING: {name} {pos} is inside a wall!")
        print(f"  Snapped to nearest free cell: {snapped}")
        return snapped
    return pos


def compute_heuristic_vector(grid, goal):
    """
    Pre-compute real distance from every free cell to goal.
    Uses 8-directional BFS — matches A* movement exactly.
    Much better than straight-line Euclidean (ignores walls).
    """
    print("\n--- Heuristic Vector ---")
    H, W = grid.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    dist[goal] = 0.0
    q = deque([goal])
    dirs = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
            (1,1,1.4),(1,-1,1.4),(-1,1,1.4),(-1,-1,1.4)]
    while q:
        r, c = q.popleft()
        for dr, dc, cost in dirs:
            nr, nc = r+dr, c+dc
            if not (0<=nr<H and 0<=nc<W): continue
            if grid[nr,nc] == 0: continue
            nd = dist[r,c] + cost
            if nd < dist[nr,nc]:
                dist[nr,nc] = nd
                q.append((nr,nc))
    reachable = np.sum(np.isfinite(dist))
    print(f"Reachable cells: {reachable} / {np.sum(grid==1)}")
    print(f"Max distance   : {np.max(dist[np.isfinite(dist)]):.1f} cells")
    return dist


def run_astar(grid, start, goal, heuristic_map, diagonals=False):
    """Run A* using astar.py with given settings."""
    planner                   = Astar()
    planner.map               = grid
    planner.goal_cell         = PosePixels(goal[0], goal[1])
    planner.heuristic_map     = heuristic_map
    planner.include_diagonals = diagonals
    planner.verbose           = False

    raw_rev = planner.run_astar(
        PosePixels(start[0], start[1]),
        PosePixels(goal[0],  goal[1])
    )
    if raw_rev is None:
        return None, planner
    return raw_rev[::-1], planner


def merge_straight(path):
    """Remove redundant waypoints on straight segments (cardinal mode)."""
    if path is None or len(path) < 3:
        return path
    merged = [path[0]]
    for i in range(1, len(path)-1):
        prev = path[i-1]; curr = path[i]; nxt = path[i+1]
        if ((curr.r-prev.r, curr.c-prev.c) !=
            (nxt.r-curr.r,  nxt.c-curr.c)):
            merged.append(curr)
    merged.append(path[-1])
    return merged


# ── kept for backward-compat with any imports
def get_smooth_path():
    grid  = load_map(MAP_FILE)
    start = validate(grid, START, "START")
    goal  = validate(grid, GOAL,  "GOAL")
    hmap  = compute_heuristic_vector(grid, goal)
    if np.isinf(hmap[start]):
        fr    = np.argwhere(np.isfinite(hmap) & (grid==1))
        dists = np.sqrt((fr[:,0]-start[0])**2+(fr[:,1]-start[1])**2)
        start = tuple(fr[np.argmin(dists)])
    path, _ = run_astar(grid, start, goal, hmap, diagonals=False)
    smooth  = merge_straight(path)
    return grid, start, goal, smooth


# ================================================================
# ── DISCRETE MODE ───────────────────────────────────────────────
# Cardinal A* → stop-turn-go → for CMN paper compliance
# ================================================================

_rotation_error = 0.0

def _get_heading(p1, p2):
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def _heading_diff(h1, h2):
    d = h2 - h1
    while d >  180: d -= 360
    while d < -180: d += 360
    return d

def _turns_needed(angle_deg):
    """Break turn into DEGREES_PER_TURN increments + error correction."""
    global _rotation_error
    n = int(round(abs(angle_deg) / DEGREES_PER_TURN))
    if n == 0: return []
    direction = 'turn_left' if angle_deg > 0 else 'turn_right'
    error = abs(angle_deg) - n * DEGREES_PER_TURN
    _rotation_error += error
    if _rotation_error >= DEGREES_PER_TURN:
        n += 1
        _rotation_error -= DEGREES_PER_TURN
    return [direction] * n

def _cells_to_steps(p1, p2):
    return max(1, round(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) * STEPS_PER_CELL))

def path_to_discrete_actions(smooth_path, start_facing=0.0):
    """Convert path waypoints to forward/turn_left/turn_right commands."""
    global _rotation_error
    _rotation_error = 0.0
    actions = []
    heading = start_facing
    pts = [(int(p.r), int(p.c)) if hasattr(p,'r') else p for p in smooth_path]
    TURN_DUR = math.radians(DEGREES_PER_TURN) / ANGULAR_MAX
    STEP_DUR = STEP_SIZE_M / LINEAR_VEL

    for i in range(len(pts)-1):
        p1,p2 = pts[i], pts[i+1]
        target  = _get_heading(p1, p2)
        turn_a  = _heading_diff(heading, target)
        for t in _turns_needed(turn_a):
            actions.append({'action':t,'steps':1,'from':p1,'to':p1,
                            'angle':DEGREES_PER_TURN,'dist_m':0.0,
                            'dur_s':round(TURN_DUR,2)})
            heading += DEGREES_PER_TURN if t=='turn_left' else -DEGREES_PER_TURN
        steps = _cells_to_steps(p1, p2)
        actions.append({'action':'move_forward','steps':steps,
                        'from':p1,'to':p2,'angle':0,
                        'dist_m':round(steps*STEP_SIZE_M,2),
                        'dur_s':round(steps*STEP_DUR,2)})
    return actions

def merge_discrete_actions(actions):
    if not actions: return actions
    merged=[]; i=0
    while i < len(actions):
        a=actions[i]; tot_s=a['steps']; tot_d=a['dist_m']; tot_t=a['dur_s']
        j=i+1
        while j<len(actions) and actions[j]['action']==a['action']:
            tot_s+=actions[j]['steps']; tot_d+=actions[j]['dist_m']
            tot_t+=actions[j]['dur_s']; j+=1
        merged.append({**a,'steps':tot_s,'to':actions[j-1]['to'],
                        'dist_m':round(tot_d,2),'dur_s':round(tot_t,2)})
        i=j
    return merged

def save_discrete_csv(actions):
    with open(CSV_DISCRETE,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['#','action','steps','angle','dist_m','dur_s','from','to'])
        for i,a in enumerate(actions):
            deg = (f"+{a['angle']}°" if 'left' in a['action'] else
                   f"-{a['angle']}°" if 'right' in a['action'] else '-')
            w.writerow([i+1,a['action'],a['steps'],deg,
                        a['dist_m'],a['dur_s'],a['from'],a['to']])
    print(f"Saved: {CSV_DISCRETE}  ({len(actions)} actions)")

def visualize_discrete(grid, start, goal, astar_path, smooth_path, actions):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ax = axes[0]
    ax.imshow(grid, cmap="gray", origin="upper", interpolation="nearest")
    px=[p.c if hasattr(p,'c') else p[1] for p in astar_path]
    py=[p.r if hasattr(p,'r') else p[0] for p in astar_path]
    sx=[p.c if hasattr(p,'c') else p[1] for p in smooth_path]
    sy=[p.r if hasattr(p,'r') else p[0] for p in smooth_path]
    ax.plot(px,py,'b--',linewidth=1,alpha=0.4,label=f'A* ({len(astar_path)} pts)')
    ax.plot(sx,sy,color='orangered',linewidth=2.5,label=f'Path ({len(smooth_path)} pts)')
    ax.scatter(sx,sy,c='orangered',s=20,zorder=5)
    for i in range(len(smooth_path)-1):
        p1=smooth_path[i]; p2=smooth_path[i+1]
        c1=(p1.c if hasattr(p1,'c') else p1[1])
        r1=(p1.r if hasattr(p1,'r') else p1[0])
        c2=(p2.c if hasattr(p2,'c') else p2[1])
        r2=(p2.r if hasattr(p2,'r') else p2[0])
        ax.annotate("",xy=(c2,r2),xytext=(c1,r1),
            arrowprops=dict(arrowstyle="-|>",color="dodgerblue",lw=1.5,mutation_scale=12))
    ax.scatter(start[1],start[0],c='lime',s=250,marker='*',zorder=8,
               edgecolors='black',label='Start')
    ax.scatter(goal[1], goal[0], c='red', s=250,marker='*',zorder=8,
               edgecolors='black',label='Goal')
    ax.legend(fontsize=8)
    fwd=sum(1 for a in actions if a['action']=='move_forward')
    turns=len(actions)-fwd
    ax.set_title(f"Discrete Mode — Cardinal A*\nForward: {fwd}  Turns: {turns}  "
                 f"deg/turn: {DEGREES_PER_TURN}°  "
                 f"{'ROS2 ✓' if ROS2_AVAILABLE else 'No ROS2'}",fontsize=11)

    ax=axes[1]; ax.axis('off')
    MAX_ROWS=20; show=actions[:MAX_ROWS]
    total_d=sum(a['dist_m'] for a in actions)
    rows=[['#','Action','Steps','Angle','Dist(m)','Dur(s)','From','To']]
    for i,a in enumerate(show):
        deg=(f"+{a['angle']}°" if 'left' in a['action'] else
             f"-{a['angle']}°" if 'right' in a['action'] else '-')
        rows.append([str(i+1),a['action'].replace('_',' '),str(a['steps']),
                     deg,str(a['dist_m']) if a['dist_m']>0 else '-',
                     str(a['dur_s']),str(a['from']),str(a['to'])])
    if len(actions)>MAX_ROWS:
        rows.append(['...', f'{len(actions)-MAX_ROWS} more','','','','',
                     f'→ {CSV_DISCRETE}',''])
    col_w=[0.04,0.15,0.07,0.08,0.08,0.07,0.15,0.15]
    table=ax.table(cellText=rows[1:],colLabels=rows[0],
                   colWidths=col_w,loc='center',cellLoc='center')
    table.auto_set_font_size(False)
    n=len(rows)-1
    table.set_fontsize(max(6,min(8,int(160/n))))
    table.scale(1,max(1.0,min(1.3,50/n)))
    for j in range(len(rows[0])):
        table[0,j].set_facecolor('#2c3e50')
        table[0,j].set_text_props(color='white',fontweight='bold')
    for i in range(1,len(rows)):
        c='#d5f5e3' if 'forward' in rows[i][1] else '#fdebd0'
        for j in range(len(rows[0])): table[i,j].set_facecolor(c)
    ax.set_title(f"Action Sequence\nTotal: {len(actions)}  Dist: {total_d:.1f}m  "
                 f"{DEGREES_PER_TURN}°/turn  {TURNS_FOR_90} turns=90°  "
                 f"→ {CSV_DISCRETE}",fontsize=9)
    plt.suptitle(f"Discrete Planner  |  Start:{start} → Goal:{goal}\n"
                 f"Cardinal A* → stop-turn-go → {DEGREES_PER_TURN}° per command",
                 fontsize=12,fontweight='bold')
    plt.savefig(OUTPUT_DISCRETE,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DISCRETE}")


# ================================================================
# ── SMOOTH MODE ─────────────────────────────────────────────────
# Diagonal A* → Pure Pursuit → for real Vinebot belt robot
# ================================================================

def _find_lookahead(pos, path, la_dist):
    dists   = [math.hypot(p[0]-pos[0], p[1]-pos[1]) for p in path]
    closest = int(np.argmin(dists))
    acc = 0.0
    for i in range(closest, len(path)-1):
        seg = math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        if acc + seg >= la_dist:
            t = (la_dist-acc)/seg
            return (path[i][0]+t*(path[i+1][0]-path[i][0]),
                    path[i][1]+t*(path[i+1][1]-path[i][1]))
        acc += seg
    return path[-1]

def simulate_smooth(grid, path):
    """
    Pure Pursuit simulation with adaptive lookahead.
    Wide corridors  → long lookahead, moves freely → curves
    Tight corridors → short lookahead, follows path → straight
    """
    H,W      = grid.shape
    MAX_TURN = math.radians(MAX_TURN_DEG)
    STEP     = 0.2
    TIGHT    = 2.0
    cl_map   = distance_transform_edt(grid)

    dr=path[1][0]-path[0][0]; dc=path[1][1]-path[0][1]
    mg=math.hypot(dr,dc); hr=dr/mg; hc=dc/mg
    pos=[float(path[0][0]),float(path[0][1])]
    traj=[tuple(pos)]; steers=[]

    for _ in range(20000):
        if math.hypot(pos[0]-path[-1][0],pos[1]-path[-1][1])<1.0: break
        ri_c=max(0,min(H-1,int(round(pos[0])))); ci_c=max(0,min(W-1,int(round(pos[1]))))
        cl=cl_map[ri_c,ci_c]
        la=LOOKAHEAD if cl>TIGHT else 1.5
        lp=_find_lookahead(tuple(pos),path,la)
        lr=lp[0]-pos[0]; lc=lp[1]-pos[1]; ld=math.hypot(lr,lc)
        if ld<0.001: break
        dr_d=lr/ld; dc_d=lc/ld
        cross=hr*dc_d-hc*dr_d; dot=hr*dr_d+hc*dc_d
        angle=math.atan2(cross,max(dot,0.001))
        turn=max(-MAX_TURN,min(MAX_TURN,angle))
        steers.append(math.degrees(turn))
        ct=math.cos(turn); st=math.sin(turn)
        new_hr=hr*ct-hc*st; new_hc=hr*st+hc*ct
        mg2=math.hypot(new_hr,new_hc); hr=new_hr/mg2; hc=new_hc/mg2
        moved=False
        if cl>TIGHT:
            nr=pos[0]+hr*STEP; nc=pos[1]+hc*STEP
            ri=int(round(nr)); ci_=int(round(nc))
            if(0<=ri<H and 0<=ci_<W) and grid[ri,ci_]==1:
                pos=[nr,nc]; moved=True
        if not moved:
            nr=pos[0]+(lr/ld)*STEP; nc=pos[1]+(lc/ld)*STEP
            ri=int(round(nr)); ci_=int(round(nc))
            if(0<=ri<H and 0<=ci_<W) and grid[ri,ci_]==1:
                pos=[nr,nc]; moved=True
        if moved: traj.append(tuple(pos))
        else: break
    return traj, steers

def _regulated_velocity(steer_deg, clearance_cells):
    """Slow on sharp turns + near obstacles — Regulated Pure Pursuit."""
    MIN_VEL=0.03; vel=LINEAR_VEL
    if abs(steer_deg)>1.0:
        radius=(0.4/CELL_SIZE_M)/math.tan(math.radians(abs(steer_deg)))
        MIN_R=0.5/CELL_SIZE_M
        if radius<MIN_R: vel=vel*max(0.0,radius/MIN_R)
    PROX=1.5
    if clearance_cells<PROX: vel=vel*(clearance_cells/PROX)
    return max(MIN_VEL,vel)

def save_smooth_csv(traj, steers, grid):
    cl_map=distance_transform_edt(grid)
    H,W=grid.shape
    with open(CSV_SMOOTH,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['#','pos_r','pos_c','steering_deg','linear_vel','angular_z'])
        for i,(pos,s) in enumerate(zip(traj,steers)):
            ri=max(0,min(H-1,int(round(pos[0])))); ci=max(0,min(W-1,int(round(pos[1]))))
            cl=float(cl_map[ri,ci])
            lin=_regulated_velocity(s,cl)
            if abs(s)<0.01: ang=0.0
            else:
                r=(0.4/CELL_SIZE_M)/math.tan(math.radians(abs(s)))
                ang=(lin/CELL_SIZE_M)/r*(1 if s>0 else -1)
                ang=max(-ANGULAR_MAX,min(ANGULAR_MAX,ang))
            w.writerow([i+1,round(pos[0],2),round(pos[1],2),
                        round(s,1),round(lin,3),round(ang,4)])
    print(f"Saved: {CSV_SMOOTH}  ({len(steers)} commands)")

def visualize_smooth(grid, start, goal, astar_path, traj, steers):
    fig,axes=plt.subplots(1,2,figsize=(18,8))
    ax=axes[0]
    ax.imshow(grid,cmap="gray",origin="upper",interpolation="nearest")
    ax.plot([p[1] for p in astar_path],[p[0] for p in astar_path],
            color='gray',linewidth=0.8,alpha=0.4,label=f'A* ({len(astar_path)} cells)')
    for i in range(len(traj)-1):
        p1=traj[i]; p2=traj[i+1]
        s=steers[i] if i<len(steers) else 0
        color='lime' if abs(s)<1.5 else 'dodgerblue' if s>0 else 'orangered'
        ax.plot([p1[1],p2[1]],[p1[0],p2[0]],color=color,linewidth=2.5,zorder=5)
    ax.scatter(start[1],start[0],c='lime',s=250,marker='*',zorder=8,
               edgecolors='black',label='Start')
    ax.scatter(goal[1], goal[0], c='red', s=250,marker='*',zorder=8,
               edgecolors='black',label='Goal')
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0],color='gray',lw=0.8,label='A* path'),
        Line2D([0],[0],color='lime',lw=2.5,label='Straight'),
        Line2D([0],[0],color='dodgerblue',lw=2.5,label='Curving left'),
        Line2D([0],[0],color='orangered',lw=2.5,label='Curving right'),
    ],fontsize=8)
    dist=len(traj)*0.2*CELL_SIZE_M
    ax.set_title(f"Smooth Mode — Diagonal A* + Pure Pursuit\n"
                 f"Lookahead: {LOOKAHEAD} cells  Max turn: {MAX_TURN_DEG}°  "
                 f"Dist: {dist:.1f}m",fontsize=11)
    ax=axes[1]; ax.set_facecolor('#1a1a2e')
    xs=list(range(len(steers)))
    ax.fill_between(xs,steers,0,where=[s>=0 for s in steers],
                    color='dodgerblue',alpha=0.7,label='Curving left')
    ax.fill_between(xs,steers,0,where=[s<0  for s in steers],
                    color='orangered', alpha=0.7,label='Curving right')
    ax.plot(xs,steers,color='white',linewidth=0.8)
    ax.axhline( MAX_TURN_DEG,color='magenta',lw=1.5,linestyle='--',
               label=f'+{MAX_TURN_DEG}° limit')
    ax.axhline(-MAX_TURN_DEG,color='magenta',lw=1.5,linestyle='--')
    ax.axhline(0,color='lime',lw=0.8)
    max_s=max(abs(s) for s in steers) if steers else 0
    avg_s=sum(abs(s) for s in steers)/len(steers) if steers else 0
    ax.set_title(f"Steering Profile\nMax: {max_s:.1f}°  Avg: {avg_s:.1f}°  "
                 f"Steps: {len(steers)}",fontsize=11,color='white')
    ax.set_xlabel('Step',color='white'); ax.set_ylabel('Steering (°)',color='white')
    ax.tick_params(colors='white')
    ax.legend(fontsize=8,facecolor='#1a1a2e',labelcolor='white')
    for sp in ['bottom','left']: ax.spines[sp].set_color('white')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.suptitle(f"Smooth Planner  |  Start:{start} → Goal:{goal}\n"
                 f"Diagonal A* + Regulated Pure Pursuit  "
                 f"(use_rotate_to_heading=False)",
                 fontsize=12,fontweight='bold')
    plt.savefig(OUTPUT_SMOOTH,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_SMOOTH}")


# ================================================================
# ── MOVEMENT COST VISUALIZATION ──────────────────────────────────
# ================================================================

def visualize_movement_cost(grid, start, goal, heuristic_map):
    """Compare cardinal vs diagonal A* on the real map."""
    fig,axes=plt.subplots(1,3,figsize=(18,6))

    # Panel 1: Heuristic vector
    ax=axes[0]
    h_show=heuristic_map.copy(); h_show[np.isinf(h_show)]=0
    h_masked=np.ma.masked_where(grid==0,h_show)
    im=ax.imshow(h_masked,cmap="viridis_r",origin="upper",interpolation="nearest")
    plt.colorbar(im,ax=ax,label="Distance to goal (cells)")
    ax.scatter(goal[1], goal[0], c="red", s=200,marker="*",zorder=6,label="Goal")
    ax.scatter(start[1],start[0],c="lime",s=200,marker="*",zorder=6,label="Start")
    ax.set_title("Heuristic Vector\nReal distance around obstacles",fontsize=11)
    ax.legend(fontsize=9)

    # Panel 2: Cardinal path (cost 1.0 per step)
    path_c, _ = run_astar(grid,start,goal,heuristic_map,diagonals=False)
    if path_c:
        merged_c = merge_straight(path_c)
        cost_c   = sum(1.0 for _ in range(len(path_c)-1))
        ax=axes[1]
        ax.imshow(grid,cmap="gray",origin="upper",interpolation="nearest")
        ax.plot([p.c for p in path_c],[p.r for p in path_c],
                'b-o',markersize=2,linewidth=1.5,label=f'Raw ({len(path_c)} pts)')
        ax.plot([p.c for p in merged_c],[p.r for p in merged_c],
                'r-',linewidth=2.5,label=f'Merged ({len(merged_c)} pts)')
        ax.scatter(start[1],start[0],c='lime',s=200,marker='*',zorder=6)
        ax.scatter(goal[1], goal[0], c='red', s=200,marker='*',zorder=6)
        ax.set_title(f"Cardinal (no diagonals)\nCost 1.0/step  Total: {cost_c:.1f}",
                     fontsize=11)
        ax.legend(fontsize=9)

    # Panel 3: Diagonal path (cost √2 per diagonal step)
    path_d, _ = run_astar(grid,start,goal,heuristic_map,diagonals=True)
    if path_d:
        cost_d = 0.0
        for i in range(len(path_d)-1):
            dr=abs(path_d[i+1].r-path_d[i].r); dc=abs(path_d[i+1].c-path_d[i].c)
            cost_d += 1.4 if (dr and dc) else 1.0
        ax=axes[2]
        ax.imshow(grid,cmap="gray",origin="upper",interpolation="nearest")
        ax.plot([p.c for p in path_d],[p.r for p in path_d],
                color='orangered',linewidth=2.5,
                label=f'Diagonal ({len(path_d)} pts)')
        ax.scatter(start[1],start[0],c='lime',s=200,marker='*',zorder=6)
        ax.scatter(goal[1], goal[0], c='red', s=200,marker='*',zorder=6)
        ax.set_title(f"Diagonal (√2=1.4 cost)\nTotal: {cost_d:.1f}  "
                     f"Saving: {((len(path_c or[])-1)-cost_d):.1f}",fontsize=11)
        ax.legend(fontsize=9)

    plt.suptitle(f"Movement Cost Comparison  |  Start:{start} → Goal:{goal}",
                 fontsize=13,fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_VIZ,dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_VIZ}")


# ================================================================
# ── ROS2 EXECUTION ───────────────────────────────────────────────
# Regulated Pure Pursuit: slows on turns + near obstacles
# ================================================================

def ros2_execute(grid, path, traj, steers):
    cl_map=distance_transform_edt(grid); H,W=grid.shape
    rclpy.init()
    node=rclpy.create_node('cmn_pipeline')
    cmd_pub =node.create_publisher(TwistStamped,CMD_TOPIC,10)
    path_pub=node.create_publisher(Path,'/plan',10)
    map_pub =node.create_publisher(OccupancyGrid,'/map',10)
    tf_static=StaticTransformBroadcaster(node)
    tf_br    =TransformBroadcaster(node)

    # map → odom static TF
    t=TransformStamped()
    t.header.stamp=node.get_clock().now().to_msg()
    t.header.frame_id='map'; t.child_frame_id='odom'
    t.transform.rotation.w=1.0; tf_static.sendTransform(t)

    # Publish map
    m=OccupancyGrid()
    m.header.stamp=node.get_clock().now().to_msg()
    m.header.frame_id='map'
    m.info.resolution=CELL_SIZE_M; m.info.width=W; m.info.height=H
    m.info.origin.orientation.w=1.0
    m.data=[0 if grid[r,c]==1 else 100
            for r in range(H) for c in range(W)]
    map_pub.publish(m)

    # Publish path
    p=Path(); p.header.stamp=node.get_clock().now().to_msg()
    p.header.frame_id='map'
    for i,cell in enumerate(path):
        ps=PoseStamped(); ps.header=p.header
        ps.pose.position.x=cell[1]*CELL_SIZE_M
        ps.pose.position.y=cell[0]*CELL_SIZE_M
        p.poses.append(ps)
    path_pub.publish(p)

    node.get_logger().info(f"Executing {len(steers)} steps — Regulated Pure Pursuit")
    STEP_TIME=0.1

    for i,(pos,s) in enumerate(zip(traj,steers)):
        ri=max(0,min(H-1,int(round(pos[0])))); ci=max(0,min(W-1,int(round(pos[1]))))
        cl=float(cl_map[ri,ci])
        lin=_regulated_velocity(s,cl)
        ang=0.0
        if abs(s)>0.01:
            r=(0.4/CELL_SIZE_M)/math.tan(math.radians(abs(s)))
            ang=(lin/CELL_SIZE_M)/r*(1 if s>0 else -1)
            ang=max(-ANGULAR_MAX,min(ANGULAR_MAX,ang))

        # odom → base_link TF
        tf=TransformStamped()
        tf.header.stamp=node.get_clock().now().to_msg()
        tf.header.frame_id='odom'; tf.child_frame_id='base_link'
        tf.transform.translation.x=pos[1]*CELL_SIZE_M
        tf.transform.translation.y=pos[0]*CELL_SIZE_M
        tf.transform.rotation.w=1.0; tf_br.sendTransform(tf)

        msg=TwistStamped()
        msg.header.stamp=node.get_clock().now().to_msg()
        msg.header.frame_id='base_footprint'
        msg.twist.linear.x=lin; msg.twist.angular.z=ang
        cmd_pub.publish(msg)

        if i%20==0:
            node.get_logger().info(
                f"  [{i}/{len(steers)}] vel={lin:.3f} steer={s:.1f}° cl={cl:.1f}")
        rclpy.spin_once(node,timeout_sec=STEP_TIME)

    stop=TwistStamped()
    stop.header.stamp=node.get_clock().now().to_msg()
    stop.header.frame_id='base_footprint'
    cmd_pub.publish(stop)
    node.get_logger().info("COMPLETE ✓")
    node.destroy_node(); rclpy.shutdown()


# ================================================================
# ── MAIN ────────────────────────────────────────────────────────
# ================================================================

def main():
    print("=" * 55)
    print(f"CMN PLANNING PIPELINE  |  MODE: {MODE.upper()}")
    print("=" * 55)
    print(f"Map          : {MAP_FILE}")
    print(f"Start / Goal : {START} → {GOAL}")
    print(f"ROS2         : {ROS2_AVAILABLE}")
    print("=" * 55)

    # Load + validate
    grid  = load_map(MAP_FILE)
    start = validate(grid, START, "START")
    goal  = validate(grid, GOAL,  "GOAL")

    # Heuristic vector (always needed)
    hmap = compute_heuristic_vector(grid, goal)
    if np.isinf(hmap[start]):
        fr    = np.argwhere(np.isfinite(hmap) & (grid==1))
        dists = np.sqrt((fr[:,0]-start[0])**2+(fr[:,1]-start[1])**2)
        start = tuple(fr[np.argmin(dists)])
        print(f"Start snapped: {start}")

    # ── Visualization mode ───────────────────────────────────────
    if MODE == "viz":
        print("\nRunning movement cost visualization...")
        visualize_movement_cost(grid, start, goal, hmap)
        print("Status: COMPLETE ✓")
        return

    # ── Discrete mode ────────────────────────────────────────────
    if MODE == "discrete":
        print("\nRunning cardinal A* (discrete mode)...")
        path, planner = run_astar(grid, start, goal, hmap, diagonals=False)
        if path is None:
            print("No path found!"); return
        smooth = merge_straight(path)
        print(f"A* path      : {len(path)} cells")
        print(f"Merged       : {len(smooth)} waypoints")

        init_h = _get_heading(
            (smooth[0].r,smooth[0].c) if hasattr(smooth[0],'r') else smooth[0],
            (smooth[1].r,smooth[1].c) if hasattr(smooth[1],'r') else smooth[1])
        actions = merge_discrete_actions(
                    path_to_discrete_actions(smooth, init_h))

        fwd   = sum(1 for a in actions if a['action']=='move_forward')
        turns = len(actions)-fwd
        total = sum(a['dist_m'] for a in actions)
        print(f"\n--- Summary ---")
        print(f"Actions      : {len(actions)}")
        print(f"Forward      : {fwd}  Turns: {turns}")
        print(f"Total dist   : {total:.2f}m")
        print(f"Deg/turn     : {DEGREES_PER_TURN}°  ({TURNS_FOR_90} turns = 90°)")

        save_discrete_csv(actions)
        visualize_discrete(grid, start, goal, path, smooth, actions)

        if ROS2_AVAILABLE:
            print("\nDiscrete mode: send commands via execute_discrete()")
        print("Status: COMPLETE ✓")
        return

    # ── Smooth mode ──────────────────────────────────────────────
    if MODE == "smooth":
        print("\nRunning diagonal A* (smooth mode)...")
        path, _ = run_astar(grid, start, goal, hmap, diagonals=True)
        if path is None:
            print("No path found!"); return
        path_tuples = [(int(p.r),int(p.c)) for p in path]
        print(f"A* path      : {len(path_tuples)} cells")

        print("Simulating Pure Pursuit...")
        traj, steers = simulate_smooth(grid, path_tuples)
        dist_goal = math.hypot(traj[-1][0]-goal[0],traj[-1][1]-goal[1])
        curved    = sum(1 for s in steers if abs(s)>1.5)
        max_s     = max(abs(s) for s in steers) if steers else 0

        print(f"\n--- Summary ---")
        print(f"Simulated    : {len(traj)} steps")
        print(f"Curved       : {curved} ({100*curved//max(len(steers),1)}%)")
        print(f"Max steering : {max_s:.1f}°")
        print(f"Dist to goal : {dist_goal:.2f} cells")

        save_smooth_csv(traj, steers, grid)
        visualize_smooth(grid, start, goal, path_tuples, traj, steers)

        if ROS2_AVAILABLE:
            print("\n" + "="*55)
            print("ROS2 ready — Regulated Pure Pursuit")
            print("  Always moves while turning (protects belts)")
            print("  Slows on sharp turns + near obstacles")
            print("="*55)
            confirm = input("Type 'yes' to execute on Vinebot: ")
            if confirm.strip().lower() == 'yes':
                ros2_execute(grid, path_tuples, traj, steers)
            else:
                print("Skipped — visualization only")
        else:
            print("\nNo ROS2 — run: conda activate habitat_ros")

        print("Status: COMPLETE ✓")
        return

    print(f"Unknown MODE: {MODE}")
    print("Set MODE = 'discrete' | 'smooth' | 'viz'")


if __name__ == "__main__":
    main()
