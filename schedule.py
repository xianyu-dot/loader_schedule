import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
try:
    with open('map_data.json', 'r', encoding='utf-8') as f:
        map_data = json.load(f)
except FileNotFoundError:
    print("Error: map_data.json not found in the current directory.")
    exit()

# Constants
PATH_TIME = 3.0             # 路径转移固定耗时 3.0 秒
CONSUMPTION_RATE = 0.5 / 60 
LOAD_TIME = 30.0            
UNLOAD_TIME = 30.0          
TURNAROUND_PENALTY = 20.0   
SWITCH_SITE_PENALTY = 60.0  

global SIM_SPEEDUP
SIM_SPEEDUP = 5             
FPS = 20                    
DT_BASE = 1.0 / FPS         

# ==========================================
# 2. Helper Functions
# ==========================================
def compute_bezier_path(p0, theta0, p1, theta1, num_points=30):
    dist = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
    control_dist = dist / 3.0
    
    c1 = (p0[0] + control_dist * math.cos(theta0), p0[1] + control_dist * math.sin(theta0))
    c2 = (p1[0] - control_dist * math.cos(theta1), p1[1] - control_dist * math.sin(theta1))
    
    path = []
    for t in np.linspace(0, 1, num_points):
        x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * c1[0] + 3*(1-t) * t**2 * c2[0] + t**3 * p1[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * c1[1] + 3*(1-t) * t**2 * c2[1] + t**3 * p1[1]
        path.append((x, y))
    return path

pts = {p['description']: (p['x'], p['y'], p['theta']) for p in map_data['points']}
REPO_PTS = {1: 'repo_1', 2: 'repo_2', 3: 'repo_3', 4: 'repo_4', 5: 'repo_5', 6: 'repo_6'}

def get_dir_category(theta):
    deg = math.degrees(theta) % 360
    if 90 < deg < 270: return 'LEFT_FACING'
    else: return 'RIGHT_FACING'

def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

# ==========================================
# 3. Classes & Dispatcher
# ==========================================
class Hopper:
    def __init__(self, id, is_left):
        self.id = id
        self.level = 20.0  
        self.is_left = is_left
        self.stats = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0} 
        
    def get_sequence(self):
        if self.is_left: return [1, 1, 2, 3]
        else: return [4, 4, 5, 6]

class Vehicle:
    def __init__(self):
        self.x, self.y, self.theta = pts['repo_1']
        self.state = "IDLE"
        self.path = []
        self.timer = 0.0
        self.current_task = None
        self.is_left_side = True
        
        self.current_pos_name = 'repo_1'
        self.needs_turnaround_flag = False
        self.final_repo = None
        self.next_state = None
        
        # 中转点记录
        self.intermediate_unload_hp_name = None
        self.final_unload_hp_name = None
        self.intermediate_repo_prep_name = None
        self.final_repo_name = None
        
        self.turn_start_theta = 0.0
        self.turn_target_theta = 0.0
        
class Dispatcher:
    def __init__(self):
        self.hoppers = {i: Hopper(i, i<=3) for i in range(1, 7)}
        self.vehicle = Vehicle()
        self.global_queue = []
        self.starving_mode = False
        self.time_elapsed = 0.0

    def update(self, dt):
        self.time_elapsed += dt
        
        for h in self.hoppers.values():
            h.level = max(0, h.level - CONSUMPTION_RATE * dt)

        starving_hoppers = [h for h in self.hoppers.values() if h.level < 15.0]
        if starving_hoppers and not self.starving_mode:
            self.starving_mode = True
            self.global_queue.clear()
            h = min(starving_hoppers, key=lambda x: x.level) 
            for s in h.get_sequence():
                self.global_queue.append((h.id, s))

        if self.starving_mode and not starving_hoppers:
            self.starving_mode = False

        if not self.global_queue and self.vehicle.state == "IDLE":
            valid_hoppers = [h for h in self.hoppers.values() if h.level <= 40.0]
            if valid_hoppers:
                best_h = min(valid_hoppers, key=lambda x: x.level)
                for s in best_h.get_sequence():
                    self.global_queue.append((best_h.id, s))

        v = self.vehicle
        if v.state == "IDLE" and self.global_queue:
            v.current_task = self.global_queue.pop(0)
            target_repo_name = REPO_PTS[v.current_task[1]]
            target_repo = pts[target_repo_name]
            target_is_left = v.current_task[0] <= 3
            
            if v.is_left_side != target_is_left:
                v.timer += SWITCH_SITE_PENALTY
                v.is_left_side = target_is_left

            # ==========================================================
            # 新增运动学校验：目标铲料点是否在车身正后方？
            # ==========================================================
            vec_f_x, vec_f_y = math.cos(v.theta), math.sin(v.theta)
            vec_t_x, vec_t_y = target_repo[0] - v.x, target_repo[1] - v.y
            dot_product = vec_f_x * vec_t_x + vec_f_y * vec_t_y

            if dot_product < 0:
                # 目标铲料点在车身后方，必须先倒车到一个更后方的卸料点作为中转
                intermediate_name = None
                min_dist = float('inf')
                
                for cand_name, cand_hp in pts.items():
                    # 仅寻找同向的后方卸料点
                    if 'hopper' in cand_name and get_dir_category(cand_hp[2]) == get_dir_category(v.theta):
                        c_vec_x, c_vec_y = cand_hp[0] - target_repo[0], cand_hp[1] - target_repo[1]
                        c_dot = vec_f_x * c_vec_x + vec_f_y * c_vec_y
                        
                        # 确保候选点位于目标铲料点的更后方
                        if c_dot < 0:
                            dist = math.hypot(cand_hp[0] - v.x, cand_hp[1] - v.y)
                            if dist < min_dist:
                                min_dist, intermediate_name = dist, cand_name

                if intermediate_name:
                    v.intermediate_repo_prep_name = intermediate_name
                    v.final_repo_name = target_repo_name
                    inter_hp = pts[intermediate_name]
                    v.path = compute_bezier_path((v.x, v.y), v.theta, inter_hp, inter_hp[2])
                    v.timer += PATH_TIME
                    v.state = "REVERSING_TO_REPO_PREP"
                    return # 阻断下方常规前进逻辑，优先执行倒车
            
            # 正常前进逻辑（目标在车前方）
            v.final_repo_name = target_repo_name
            v.final_repo = target_repo
            v.needs_turnaround_flag = False
            
            if get_dir_category(v.theta) != get_dir_category(target_repo[2]):
                v.needs_turnaround_flag = True
                
            v.path = compute_bezier_path((v.x, v.y), v.theta, target_repo, target_repo[2])
            v.timer += PATH_TIME
            v.state = "MOVING_TO_REPO"

        # ----------------新增倒车备铲状态----------------
        elif v.state == "REVERSING_TO_REPO_PREP":
            v.timer -= dt
            if v.timer <= 0:
                inter_hp = pts[v.intermediate_repo_prep_name]
                v.x, v.y, v.theta = inter_hp
                
                final_repo = pts[v.final_repo_name]
                v.final_repo = final_repo
                
                # 倒车完成，准备前进驶入铲料点。若朝向不符仍需掉头
                if get_dir_category(v.theta) != get_dir_category(final_repo[2]):
                    v.state = "TURNAROUND"
                    v.timer = TURNAROUND_PENALTY
                    v.turn_start_theta = v.theta
                    v.turn_target_theta = final_repo[2]
                    v.next_state = "MOVING_TO_FINAL_REPO"
                else:
                    v.path = compute_bezier_path((v.x, v.y), v.theta, final_repo, final_repo[2])
                    v.timer = PATH_TIME
                    v.state = "MOVING_TO_FINAL_REPO"
        # ----------------------------------------------

        elif v.state == "MOVING_TO_REPO":
            v.timer -= dt
            if v.timer <= 0:
                v.x, v.y = v.final_repo[0], v.final_repo[1]
                if v.needs_turnaround_flag:
                    v.state = "TURNAROUND"
                    v.timer = TURNAROUND_PENALTY
                    v.turn_start_theta = v.theta
                    v.turn_target_theta = v.final_repo[2]
                    v.next_state = "LOADING"
                else:
                    v.state = "LOADING"
                    v.timer = LOAD_TIME
                    v.theta = v.final_repo[2]

        elif v.state == "TURNAROUND":
            v.timer -= dt
            progress = max(0.0, min(1.0, 1.0 - (v.timer / TURNAROUND_PENALTY)))
            angle_diff = wrap_angle(v.turn_target_theta - v.turn_start_theta)
            v.theta = v.turn_start_theta + angle_diff * progress
            
            if v.timer <= 0:
                v.theta = v.turn_target_theta
                if v.next_state == "MOVING_TO_FINAL_REPO":
                    v.path = compute_bezier_path((v.x, v.y), v.theta, v.final_repo, v.final_repo[2])
                    v.timer = PATH_TIME
                    v.state = "MOVING_TO_FINAL_REPO"
                else:
                    v.state = "LOADING"
                    v.timer = LOAD_TIME

        elif v.state == "MOVING_TO_FINAL_REPO":
            v.timer -= dt
            if v.timer <= 0:
                v.x, v.y, v.theta = v.final_repo
                v.state = "LOADING"
                v.timer = LOAD_TIME

        elif v.state == "LOADING":
            v.timer -= dt
            if v.timer <= 0:
                h_id, s_id = v.current_task
                target_hp_name = None

                # 1. 前瞻优化: 后排料仓退出选择
                if s_id in [3, 6] and self.global_queue:
                    next_task = self.global_queue[0]
                    next_repo = pts[REPO_PTS[next_task[1]]]
                    hp_l_name, hp_r_name = f'hopper_left_{h_id}', f'hopper_right_{h_id}'
                    if hp_l_name in pts and hp_r_name in pts:
                        l_turn = get_dir_category(pts[hp_l_name][2]) != get_dir_category(next_repo[2])
                        r_turn = get_dir_category(pts[hp_r_name][2]) != get_dir_category(next_repo[2])
                        if l_turn and not r_turn: target_hp_name = hp_r_name
                        elif r_turn and not l_turn: target_hp_name = hp_l_name

                # 2. 同向匹配兜底
                if not target_hp_name:
                    deg = math.degrees(v.theta) % 360
                    target_hp_name = f'hopper_right_{h_id}' if (90 < deg < 270) else f'hopper_left_{h_id}'
                    if target_hp_name not in pts:
                        target_hp_name = f'hopper_left_{h_id}' if f'hopper_left_{h_id}' in pts else f'hopper_right_{h_id}'
                
                # 3. 卸料空间探测：避免直接穿模前进
                hp = pts[target_hp_name]
                vec_f_x, vec_f_y = math.cos(v.theta), math.sin(v.theta) 
                vec_t_x, vec_t_y = hp[0] - v.x, hp[1] - v.y             
                dot_product = vec_f_x * vec_t_x + vec_f_y * vec_t_y
                
                if dot_product > 0:
                    prefix = target_hp_name.rsplit('_', 1)[0] + '_'
                    intermediate_name = None
                    min_dist = float('inf')
                    
                    for i in range(1, 7):
                        cand_name = f"{prefix}{i}"
                        if cand_name in pts:
                            c_hp = pts[cand_name]
                            c_vec_x, c_vec_y = c_hp[0] - v.x, c_hp[1] - v.y
                            c_dot = vec_f_x * c_vec_x + vec_f_y * c_vec_y
                            if c_dot < 0: 
                                dist = math.hypot(c_vec_x, c_vec_y)
                                if dist < min_dist:
                                    min_dist, intermediate_name = dist, cand_name

                    if intermediate_name:
                        v.intermediate_unload_hp_name = intermediate_name
                        v.final_unload_hp_name = target_hp_name
                        inter_hp = pts[intermediate_name]
                        v.path = compute_bezier_path((v.x, v.y), v.theta, inter_hp, inter_hp[2])
                        v.timer = PATH_TIME
                        v.state = "REVERSING_OUT"
                        return 
                
                v.current_pos_name = target_hp_name
                v.path = compute_bezier_path((v.x, v.y), v.theta, hp, hp[2])
                v.timer = PATH_TIME
                v.state = "MOVING_TO_HOPPER"

        elif v.state == "REVERSING_OUT":
            v.timer -= dt
            if v.timer <= 0:
                inter_hp = pts[v.intermediate_unload_hp_name]
                v.x, v.y, v.theta = inter_hp
                
                v.current_pos_name = v.final_unload_hp_name
                final_hp = pts[v.final_unload_hp_name]
                v.path = compute_bezier_path((v.x, v.y), v.theta, final_hp, final_hp[2])
                v.timer = PATH_TIME
                v.state = "MOVING_TO_HOPPER"

        elif v.state == "MOVING_TO_HOPPER":
            v.timer -= dt
            if v.timer <= 0:
                hp = pts[v.current_pos_name]
                v.x, v.y, v.theta = hp
                v.state = "UNLOADING"
                v.timer = UNLOAD_TIME

        elif v.state == "UNLOADING":
            v.timer -= dt
            if v.timer <= 0:
                h_id, s_id = v.current_task
                self.hoppers[h_id].level = min(50.0, self.hoppers[h_id].level + 10.0)
                self.hoppers[h_id].stats[s_id] += 1
                v.state = "IDLE"
                v.current_task = None
                v.path = []

# ==========================================
# 4. Visualization Setup
# ==========================================
fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(right=0.75, bottom=0.15)
dispatcher = Dispatcher()
is_paused = False

def toggle_pause(event):
    global is_paused
    if event.inaxes != ax_speed_up and event.inaxes != ax_speed_dn:
        is_paused = not is_paused
fig.canvas.mpl_connect('button_press_event', toggle_pause)

for r in map_data['regions']:
    poly = patches.Polygon(r['boundary'], closed=True, alpha=0.3, 
                           color='gray' if r['type'] == '可行驶区域' else 
                                 'orange' if '料堆' in r['type'] else 'blue')
    ax.add_patch(poly)
    if 'num' in r:
        cx = sum(p[0] for p in r['boundary'])/4
        cy = sum(p[1] for p in r['boundary'])/4
        lbl = f"Stockpile {r['num']}" if '料堆' in r['type'] else f"Hopper {r['num']}"
        ax.text(cx, cy, lbl, ha='center', va='center', fontsize=9, weight='bold')

time_text = ax.text(-1000, 5200, "", fontsize=12, weight='bold')
state_text = ax.text(-1000, 4600, "", fontsize=11, color='darkred', weight='bold')
queue_text = fig.text(0.78, 0.55, "", fontsize=10, va='center', family='monospace')

ax_speed_up = plt.axes([0.86, 0.05, 0.08, 0.05])
ax_speed_dn = plt.axes([0.76, 0.05, 0.08, 0.05])
btn_up = Button(ax_speed_up, 'Speed +')
btn_dn = Button(ax_speed_dn, 'Speed -')

def increase_speed(event):
    global SIM_SPEEDUP
    SIM_SPEEDUP = min(100, SIM_SPEEDUP + 1)
def decrease_speed(event):
    global SIM_SPEEDUP
    SIM_SPEEDUP = max(1, SIM_SPEEDUP - 1)
btn_up.on_clicked(increase_speed)
btn_dn.on_clicked(decrease_speed)

def on_key(event):
    if event.key in ['up', 'right', '+']: increase_speed(None)
    elif event.key in ['down', 'left', '-']: decrease_speed(None)
fig.canvas.mpl_connect('key_press_event', on_key)

vehicle_patch = patches.Rectangle((0, 0), 300, 150, angle=0, color='red', zorder=5)
vehicle_head = patches.Circle((0, 0), 50, color='yellow', zorder=6)
ax.add_patch(vehicle_patch)
ax.add_patch(vehicle_head)

path_line, = ax.plot([], [], 'r--', alpha=0.5)

hopper_texts = {}
for i in range(1, 7):
    hcx, hcy = 0, 0
    for r in map_data['regions']:
        if r['type'] == '漏斗范围' and r.get('num') == i:
            hcx = sum(p[0] for p in r['boundary'])/4
            hcy = sum(p[1] for p in r['boundary'])/4
            break
    hopper_texts[i] = ax.text(hcx, hcy + 600, "", color='blue', fontsize=9, ha='center')

ax.set_xlim(-1500, 8000)
ax.set_ylim(-1000, 6000)
ax.set_title("Autonomous Loader Dispatch (Click background to Pause/Resume)", fontsize=14)
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")

def update(frame):
    if not is_paused:
        dt = DT_BASE * SIM_SPEEDUP
        dispatcher.update(dt)
        
    mins = int(dispatcher.time_elapsed // 60)
    secs = int(dispatcher.time_elapsed % 60)
    time_text.set_text(f"Time: {mins:02d}:{secs:02d}")
    
    v = dispatcher.vehicle
    state_display = f"TURNAROUND ({int(v.timer)}s)" if v.state == "TURNAROUND" else v.state
    task_info = f"Load S{v.current_task[1]} -> Unload H{v.current_task[0]}" if v.current_task else "None"
    
    status_header = "[ PAUSED ]\n" if is_paused else ""
    state_text.set_text(f"{status_header}Vehicle: {state_display}\nTask: {task_info}\nSim Speed: {SIM_SPEEDUP}x")
    
    # 动画刷新加入新状态
    if v.path and v.state in ["MOVING_TO_REPO", "MOVING_TO_FINAL_REPO", "REVERSING_OUT", "REVERSING_TO_REPO_PREP", "MOVING_TO_HOPPER"]:
        idx = int((1.0 - v.timer / PATH_TIME) * len(v.path))
        idx = max(0, min(idx, len(v.path)-1))
        cx, cy = v.path[idx]
        if idx < len(v.path)-1:
            v.theta = math.atan2(v.path[idx+1][1]-cy, v.path[idx+1][0]-cx)
        path_line.set_data([p[0] for p in v.path], [p[1] for p in v.path])
    else:
        cx, cy = v.x, v.y
        path_line.set_data([], [])

    deg = math.degrees(v.theta)
    offset_x = cx - 150*math.cos(v.theta) + 75*math.sin(v.theta)
    offset_y = cy - 150*math.sin(v.theta) - 75*math.cos(v.theta)
    vehicle_patch.set_xy((offset_x, offset_y))
    vehicle_patch.angle = deg
    vehicle_head.set_center((cx + 150*math.cos(v.theta), cy + 150*math.sin(v.theta)))

    for i, h in dispatcher.hoppers.items():
        stats_str = ", ".join([f"S{k}:{v}" for k,v in h.stats.items() if v>0])
        color = 'red' if h.level < 15 else 'green' if h.level < 40 else 'blue'
        hopper_texts[i].set_text(f"H{i}: {h.level:.1f}%\nStats: [{stats_str}]")
        hopper_texts[i].set_color(color)

    q_str = "=== Task Queue ===\n\n"
    if dispatcher.starving_mode:
        q_str += "!! EMERGENCY !!\nStarving Mode Active\n\n"
    for t in dispatcher.global_queue[:12]:
        q_str += f"-> Hopper {t[0]} from S{t[1]}\n"
    if len(dispatcher.global_queue) > 12:
        q_str += "...\n"
    queue_text.set_text(q_str)

    return path_line, vehicle_patch, vehicle_head, time_text, state_text, queue_text, *hopper_texts.values()

ani = FuncAnimation(fig, update, frames=2000, interval=int(DT_BASE*1000), blit=False)
plt.show()