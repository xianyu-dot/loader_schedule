import sys
import json
import math
import re
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTableWidget, 
                             QTableWidgetItem, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QWidget, QFileDialog, QMessageBox, QLabel, QProgressDialog, QSplitter, QLineEdit)
from PyQt5.QtCore import Qt

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

# ==========================================
# 1. 運動學路徑計算引擎與公式解析器
# ==========================================
SPEED_CM_S = 200.0
REVERSE_COST = 3000.0  # 掉頭動作的標準懲罰距離

def get_path_length(path):
    if path is None or len(path) < 2: return float('inf')
    return np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))

def get_bezier_curve(p0, p3, theta0, theta3, gear, num_points=40):
    p0, p3 = np.array(p0), np.array(p3)
    dist = np.linalg.norm(p3 - p0)
    d = max(dist * 0.45, 300.0) 
    
    v0 = np.array([math.cos(theta0), math.sin(theta0)]) * gear
    v3 = np.array([math.cos(theta3), math.sin(theta3)]) * gear
        
    p1, p2 = p0 + v0 * d, p3 - v3 * d
    
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 4)) 
    for i in range(num_points):
        tt = t[i]
        pos = ((1-tt)**3 * p0 + 3*(1-tt)**2 * tt * p1 + 3*(1-tt) * tt**2 * p2 + tt**3 * p3)
        curve[i] =[pos[0], pos[1], 0, gear] 
    return curve

def parse_formula(formula_str, points, cost_formulas=None, depth=0):
    """
    核心：公式解析引擎。支持 forward, backward, reverse, route 组合，以及纯数字。
    """
    if depth > 20:
        return None, float('inf'), "Recursion Error: Formula loop detected"
        
    if formula_str is None or str(formula_str).strip().upper() == "INF":
        return None, float('inf'), "Success"

    parts =[p.strip() for p in str(formula_str).split('+')]
    full_path =[]
    total_cost = 0.0
    
    for part in parts:
        if part == 'reverse':
            total_cost += REVERSE_COST
            continue
            
        # [新增] 允许直接输入纯数字距离（例如硬性补给距离，或单独 override 为 1500）
        try:
            val = float(part)
            total_cost += val
            continue
        except ValueError:
            pass
            
        # 支持路径组合：route(A, B) 或 path(A, B)
        m_route = re.match(r'(route|path)\(([^,]+),\s*([^)]+)\)', part)
        if m_route:
            _, p1_name, p2_name = m_route.groups()
            p1_name, p2_name = p1_name.strip(), p2_name.strip()
            
            if cost_formulas is None:
                return None, float('inf'), "Error: cost_formulas dictionary missing for route parsing"
                
            sub_formula = cost_formulas.get(p1_name, {}).get(p2_name, "INF")
            sub_path, sub_cost, sub_msg = parse_formula(sub_formula, points, cost_formulas, depth + 1)
            
            if sub_msg != "Success" or sub_cost == float('inf'):
                return None, float('inf'), f"Error in sub-route {p1_name}->{p2_name}: {sub_msg}"
                
            total_cost += sub_cost
            if len(full_path) == 0:
                full_path = sub_path
            elif sub_path is not None and len(sub_path) > 0:
                full_path = np.vstack((full_path, sub_path))
            continue

        # 原有的 forward / backward 逻辑
        m = re.match(r'(forward|backward)\(([^,]+),\s*([^)]+)\)', part)
        if m:
            action, p1_name, p2_name = m.groups()
            p1_name, p2_name = p1_name.strip(), p2_name.strip()
            
            if p1_name not in points or p2_name not in points:
                return None, float('inf'), f"Error: Point not found ({p1_name} or {p2_name})"
                
            p1 = points[p1_name]
            p2 = points[p2_name]
            gear = 1 if action == 'forward' else -1
            
            curve = get_bezier_curve([p1['x'], p1['y']], [p2['x'], p2['y']], p1['theta'], p2['theta'], gear)
            length = get_path_length(curve)
            total_cost += length
            
            if len(full_path) == 0:
                full_path = curve
            else:
                full_path = np.vstack((full_path, curve))
        else:
            return None, float('inf'), f"Syntax Error near: '{part}'"
            
    return full_path, total_cost, "Success"

def generate_business_formula(n1, n2):
    # 【新增】滿足特定路徑組合要求：
    if n1 == 'hopper_left_4' and n2 == 'repo_1':
        return f"route({n1}, hopper_left_3) + route(hopper_left_3, {n2})"
    if n1 == 'hopper_left_4' and n2 == 'hopper_left_3':
        return f"backward({n1}, {n2})"

    is_h1_left = 'hopper_left' in n1
    is_h1_right = 'hopper_right' in n1
    is_r1 = 'repo' in n1
    is_h2_left = 'hopper_left' in n2
    is_h2_right = 'hopper_right' in n2
    is_r2 = 'repo' in n2

    if (is_h1_left or is_h1_right) and is_r2:
        if is_h1_left:
            if n2 in ['repo_2', 'repo_3', 'repo_5', 'repo_6']:
                return f"forward({n1}, {n2})"
            elif n2 == 'repo_1':
                return f"forward({n1}, reverse_point_1) + reverse + forward(reverse_point_1, {n2})"
            elif n2 == 'repo_4':
                return f"forward({n1}, reverse_point_2) + reverse + forward(reverse_point_2, {n2})"
                
        elif is_h1_right:
            if n2 in['repo_1', 'repo_3']:
                return f"backward({n1}, hopper_right_2) + forward(hopper_right_2, {n2})"
            elif n2 == 'repo_2':
                return f"backward({n1}, hopper_right_2) + forward(hopper_right_2, reverse_point_1) + reverse + forward(reverse_point_1, {n2})"
            elif n2 == 'repo_4':
                return f"backward({n1}, hopper_right_4) + forward(hopper_right_4, {n2})"
            elif n2 in['repo_5', 'repo_6']:
                return f"forward({n1}, {n2})"
                
    elif is_r1 and (is_h2_left or is_h2_right):
        if n1 in['repo_2', 'repo_4']:
            if is_h2_left:
                return f"backward({n1}, {n2})"
        elif n1 == 'repo_1':
            if is_h2_right:
                return f"backward({n1}, hopper_right_2) + forward(hopper_right_2, {n2})"
        elif n1 == 'repo_5':
            if is_h2_right:
                return f"backward({n1}, hopper_right_5) + forward(hopper_right_5, {n2})"
        elif n1 in['repo_3', 'repo_6']:
            zone1 = 'left' if n1 == 'repo_3' else 'right'
            try:
                num = int(n2.split('_')[-1])
                zone2 = 'left' if num <= 3 else 'right'
                if zone1 == zone2:
                    return f"backward({n1}, {n2})"
            except: pass

    return "INF"

# ==========================================
# 2. PyQt5 可视化标定工作站
# ==========================================
class MatrixStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.map_data = {}
        self.points = {}
        self.node_names =[]
        
        self.cost_matrix = {}
        self.cost_formulas = {}   
        self.cached_paths = {}        
        self.selected_pair = None     

        self.initUI()
        self.load_map_data()

    def initUI(self):
        self.setWindowTitle('Auto Loader Formula Matrix & Trajectory Studio')
        self.setGeometry(50, 50, 1600, 1000)

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        toolbar_layout = QHBoxLayout()
        self.btn_compute = QPushButton('1. Compute Business Rule Matrix')
        self.btn_compute.clicked.connect(self.compute_all_matrix)
        self.btn_compute.setStyleSheet("background-color: #2196F3; color: white; padding: 12px; font-weight: bold; border-radius: 4px;")
        
        self.btn_load = QPushButton('Load Matrix JSON')
        self.btn_load.clicked.connect(self.load_matrix_json)
        self.btn_load.setStyleSheet("padding: 12px; border-radius: 4px;")
        
        self.btn_save = QPushButton('Export Cost Matrix JSON')
        self.btn_save.clicked.connect(self.save_matrix_json)
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; padding: 12px; font-weight: bold; border-radius: 4px;")

        toolbar_layout.addWidget(self.btn_compute)
        toolbar_layout.addWidget(self.btn_load)
        toolbar_layout.addWidget(self.btn_save)

        splitter = QSplitter(Qt.Vertical)

        top_panel = QWidget()
        top_layout = QVBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_panel.setLayout(top_layout)

        formula_toolbar = QHBoxLayout()
        self.lbl_selected_route = QLabel("Route: None")
        self.lbl_selected_route.setFixedWidth(250)
        self.lbl_selected_route.setStyleSheet("font-size: 14px; font-weight: bold; color: #E91E63; padding: 5px;")
        
        formula_label = QLabel("Formula:")
        formula_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.edit_formula = QLineEdit()
        self.edit_formula.setPlaceholderText("e.g. route(A, B), forward(A, B) + 500, or just 1500")
        self.edit_formula.setStyleSheet("padding: 6px; font-size: 14px; background: #fff;")
        
        self.btn_apply_formula = QPushButton("Apply Formula Override")
        self.btn_apply_formula.clicked.connect(self.apply_custom_formula)
        self.btn_apply_formula.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 6px 15px; border-radius: 4px;")

        formula_toolbar.addWidget(self.lbl_selected_route)
        formula_toolbar.addWidget(formula_label)
        formula_toolbar.addWidget(self.edit_formula, 1) 
        formula_toolbar.addWidget(self.btn_apply_formula)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        top_layout.addLayout(formula_toolbar)
        top_layout.addWidget(self.canvas)
        
        self.table = QTableWidget()
        self.table.cellClicked.connect(self.on_cell_clicked)
        self.is_updating_table = False 

        splitter.addWidget(top_panel)
        splitter.addWidget(self.table)
        splitter.setSizes([550, 450]) 

        main_layout.addLayout(toolbar_layout)
        main_layout.addWidget(splitter, 1) 
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def load_map_data(self):
        try:
            with open('map_data.json', 'r', encoding='utf-8') as f:
                self.map_data = json.load(f)
            
            self.points = {}
            rev_count = 1
            for p in self.map_data['points']:
                desc = p['description']
                if desc == 'reverse_point':
                    desc = f"reverse_point_{rev_count}"
                    p['description'] = desc
                    rev_count += 1
                self.points[desc] = p
                
            self.node_names =[name for name in self.points.keys() if 'repo' in name or 'hopper' in name]
            self.node_names.sort()
            
            for n1 in self.node_names:
                self.cost_matrix[n1] = {}
                self.cost_formulas[n1] = {}
                for n2 in self.node_names:
                    self.cost_matrix[n1][n2] = 0.0 if n1 == n2 else float('inf')
                    self.cost_formulas[n1][n2] = "INF" if n1 != n2 else "0"
                    
            self.update_table_ui()
            self.init_map_plot()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load map_data.json: {e}")

    def init_map_plot(self):
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_title("Path Visualization with Headings")
        
        colors = {"可行駛區域": "#f5f5f5", "料堆範圍": "#ffe0b2", "漏斗範圍": "#bbdefb"}
        for r in self.map_data.get('regions',[]):
            poly = patches.Polygon(r['boundary'], closed=True, facecolor=colors.get(r['type'], "gray"), edgecolor='black', alpha=0.5)
            self.ax.add_patch(poly)
            if 'num' in r:
                cx = sum(p[0] for p in r['boundary'])/4
                cy = sum(p[1] for p in r['boundary'])/4
                tag = f"B{r['num']}" if "料堆" in r['type'] else f"H{r['num']}"
                self.ax.text(cx, cy, tag, ha='center', va='center', fontsize=8, color='#555')

        for name, p in self.points.items():
            self.ax.plot(p['x'], p['y'], 'k.', markersize=4)
            short_name = name.replace('hopper_left_', 'hl_').replace('hopper_right_', 'hr_').replace('reverse_point_', 'rev_')
            self.ax.text(p['x'], p['y'] - 120, short_name, fontsize=7, color='#333', ha='center', fontweight='bold')
            dx = 180 * math.cos(p['theta'])
            dy = 180 * math.sin(p['theta'])
            self.ax.arrow(p['x'], p['y'], dx, dy, head_width=70, head_length=90, fc='#9C27B0', ec='#9C27B0', alpha=0.7)

        self.line_path, = self.ax.plot([],[], 'g-', linewidth=3.0, alpha=0.8, label='Planned Formula Path')
        self.line_start, = self.ax.plot([],[], 'go', markersize=8, label='Start')
        self.line_end, = self.ax.plot([],[], 'rx', markersize=8, label='End')
        
        self.ax.legend(loc='upper right')
        self.figure.tight_layout()
        self.canvas.draw()

    def on_cell_clicked(self, row, col):
        n1 = self.node_names[row]
        n2 = self.node_names[col]
        self.selected_pair = (n1, n2)
        
        cost = self.cost_matrix[n1].get(n2, float('inf'))
        self.lbl_selected_route.setText(f"Route: [{n1}] ➔[{n2}]\nCost: {'INF' if cost == float('inf') else f'{cost:.1f} cm'}")
        
        formula = self.cost_formulas.get(n1, {}).get(n2, "INF")
        self.edit_formula.setText(formula)
        
        self.render_selected_route()

    def apply_custom_formula(self):
        if not self.selected_pair: return
        n1, n2 = self.selected_pair
        
        formula = self.edit_formula.text()
        path, cost, msg = parse_formula(formula, self.points, self.cost_formulas)
        
        if msg == "Success":
            self.cached_paths[(n1, n2)] = path
            self.cost_matrix[n1][n2] = cost
            self.cost_formulas[n1][n2] = formula
            
            self.update_table_ui()
            self.render_selected_route()
            self.lbl_selected_route.setText(f"Route: [{n1}] ➔ [{n2}]\nCost: {'INF' if cost == float('inf') else f'{cost:.1f} cm'}")
        else:
            QMessageBox.warning(self, "Formula Parse Error", msg)

    def render_selected_route(self):
        if not self.selected_pair: return
        n1, n2 = self.selected_pair
        p1, p2 = self.points[n1], self.points[n2]
        
        self.line_start.set_data([p1['x']], [p1['y']])
        self.line_end.set_data([p2['x']], [p2['y']])
        
        path = self.cached_paths.get((n1, n2), None)
        if path is not None and len(path) > 0:
            self.line_path.set_data(path[:, 0], path[:, 1])
        else:
            self.line_path.set_data([],[])
            
        self.canvas.draw()

    def update_table_ui(self):
        self.is_updating_table = True
        n = len(self.node_names)
        self.table.setRowCount(n)
        self.table.setColumnCount(n)
        self.table.setHorizontalHeaderLabels(self.node_names)
        self.table.setVerticalHeaderLabels(self.node_names)

        for i, row_node in enumerate(self.node_names):
            for j, col_node in enumerate(self.node_names):
                val = self.cost_matrix.get(row_node, {}).get(col_node, float('inf'))
                formula = self.cost_formulas.get(row_node, {}).get(col_node, "INF")
                
                display_text = "INF" if val == float('inf') else f"{val:.1f}"
                item = QTableWidgetItem(display_text)
                item.setTextAlignment(Qt.AlignCenter)
                
                # 【修改点 1】: 将表格设置为绝对只读，防止直接双击编辑导致的保存失败错觉
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                
                if i == j:
                    item.setBackground(Qt.lightGray)
                    item.setFlags(Qt.ItemIsEnabled) 
                elif val == float('inf'):
                    item.setBackground(Qt.darkRed)
                    item.setForeground(Qt.white)
                elif "route" in formula or "path" in formula:
                    item.setBackground(Qt.magenta) 
                    item.setForeground(Qt.white)
                elif '+' in formula or "forward" in formula or "backward" in formula:
                    if "reverse" in formula:
                        item.setBackground(Qt.yellow) 
                        item.setForeground(Qt.black)
                    else:
                        item.setBackground(Qt.cyan) 
                        item.setForeground(Qt.black)
                    
                self.table.setItem(i, j, item)
                
        self.table.resizeColumnsToContents()
        self.is_updating_table = False

    def compute_all_matrix(self):
        total = len(self.node_names) ** 2
        progress = QProgressDialog("Generating and Parsing Formulas...", "Cancel", 0, total * 2, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        step = 0
        for n1 in self.node_names:
            if n1 not in self.cost_matrix: 
                self.cost_matrix[n1] = {}
                self.cost_formulas[n1] = {}
            for n2 in self.node_names:
                if progress.wasCanceled(): return
                if n1 != n2:
                    self.cost_formulas[n1][n2] = generate_business_formula(n1, n2)
                else:
                    self.cost_formulas[n1][n2] = "0"
                step += 1
                progress.setValue(step)

        for n1 in self.node_names:
            for n2 in self.node_names:
                if progress.wasCanceled(): return
                if n1 != n2:
                    formula = self.cost_formulas[n1][n2]
                    path, cost, msg = parse_formula(formula, self.points, self.cost_formulas)
                    if msg == "Success":
                        self.cached_paths[(n1, n2)] = path
                        self.cost_matrix[n1][n2] = cost
                    else:
                        self.cost_matrix[n1][n2] = float('inf')
                else:
                    self.cost_matrix[n1][n2] = 0.0 
                step += 1
                progress.setValue(step)
                QApplication.processEvents() 
                
        self.update_table_ui()
        if self.selected_pair: 
            self.on_cell_clicked(self.node_names.index(self.selected_pair[0]), self.node_names.index(self.selected_pair[1]))

    def save_matrix_json(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Cost Matrix", "cost_matrix.json", "JSON Files (*.json)")
        if filename:
            export_data = {
                "metadata": {
                    "description": "Autonomous Loader Routing Matrix",
                    "version": "2.0"
                },
                "cost_matrix": self.cost_matrix,
                "cost_formulas": self.cost_formulas
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=4)
            QMessageBox.information(self, "Success", f"Saved matrix and formulas to {filename}")

    def load_matrix_json(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Cost Matrix", "", "JSON Files (*.json)")
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "cost_matrix" in data and "cost_formulas" in data:
                    self.cost_matrix = data["cost_matrix"]
                    self.cost_formulas = data["cost_formulas"]
                else:
                    self.cost_matrix = data
                    for n1 in self.node_names:
                        if n1 not in self.cost_formulas: self.cost_formulas[n1] = {}
                        for n2 in self.node_names:
                            val = self.cost_matrix.get(n1, {}).get(n2, float('inf'))
                            self.cost_formulas[n1][n2] = "INF" if val == float('inf') else "Loaded Legacy Value"
                            
                # 【修改点 2】: 导入时强制使用公式重算 cost，防止 JSON 里旧距离和公式不统一
                self.cached_paths.clear()
                for n1 in self.cost_formulas:
                    for n2 in self.cost_formulas[n1]:
                        formula = self.cost_formulas[n1][n2]
                        if formula not in ["INF", "0", "Loaded Legacy Value"]:
                            path, cost, msg = parse_formula(formula, self.points, self.cost_formulas)
                            if msg == "Success":
                                self.cached_paths[(n1, n2)] = path
                                self.cost_matrix[n1][n2] = cost  # 强制覆盖为实时重算的结果，彻底杜绝数据还原失败
                                
                self.update_table_ui()
                QMessageBox.information(self, "Success", f"Loaded matrix and restored formulas from {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MatrixStudio()
    ex.show()
    sys.exit(app.exec_())