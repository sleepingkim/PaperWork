# -*- coding: utf-8 -*-
# 필요한 라이브러리들을 임포트합니다.
import sys # 시스템 관련 모듈
import os # 운영체제 관련 모듈
import time # 시간 관련 모듈
import heapq # A* 알고리즘을 위한 우선순위 큐

try:
    # Ursina: 3D 게임 엔진 및 시뮬레이션 환경 제공
    from ursina import *
    # Scikit-learn: DBSCAN, K-Means, GMM 클러스터링 알고리즘 사용
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.mixture import GaussianMixture
    import hdbscan # HDBSCAN 추가
    # NumPy: 다차원 배열 및 수학 연산 지원 (필수)
    import numpy as np
    # Matplotlib: 데이터 시각화 (특히 3D 플롯)
    from mpl_toolkits.mplot3d import Axes3D # 3D 플롯 위해 추가
    import open3d as o3d # Open3D: 3D 데이터 처리 (RANSAC 등)
except ImportError as e:
    t_now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_now}] Error: Failed to import essential libraries: {e}")
    print(f"[{t_now}] Please ensure Ursina, Scikit-learn, NumPy, Pandas, Matplotlib, Open3D, and HDBSCAN are installed.")
    print(f"[{t_now}] Example install command: pip install ursina scikit-learn numpy pandas matplotlib open3d hdbscan")
    sys.exit(1)
except Exception as e:
    t_now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_now}] An unexpected error occurred during import: {e}")
    sys.exit(1)

import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pickle

# --- 시뮬레이션 환경 설정값 ---
RANDOM_OBSTACLE_SEED = 1
current_obstacle_seed = RANDOM_OBSTACLE_SEED

NUM_OBSTACLES = 80
AREA_SIZE = 100
AGENT_SPEED = 30
MANUAL_AGENT_SPEED = 30
ROTATION_SPEED = 200
AGENT_HEIGHT = 0.5
LIDAR_RANGE = 30
LIDAR_FOV = 360
NUM_LIDAR_RAYS = 180
LIDAR_VERTICAL_FOV = 45
NUM_LIDAR_CHANNELS = 8
LIDAR_VIS_HEIGHT = 0.6
LIDAR_COLOR = color.cyan
OBSTACLE_TAG = "obstacle"
SCAN_INTERVAL = 0.5
MANUAL_SCAN_INTERVAL = 1.0
VISUALIZE_LIDAR_RAYS = False

# --- 라이다 노이즈 설정 ---
LIDAR_NOISE_ENABLED = True # True로 변경하여 노이즈 활성화
LIDAR_NOISE_LEVEL = 0.05    # 각 좌표에 더해지거나 빼질 최대 노이즈 값 (월드 단위)

min_obstacle_distance_from_spawn = 3.0
max_placement_attempts_per_obstacle = 100

MIN_MOVEMENT_THRESHOLD_FOR_NEW_SCAN = 0.1
MIN_ROTATION_THRESHOLD_FOR_NEW_SCAN_DEG = 1.0

# --- A* 및 자율 주행 설정값 ---
GRID_CELL_SIZE = 1.0
OBSTACLE_PADDING_CELLS = 1
WAYPOINT_THRESHOLD = 1.5
TOTAL_SIMULATION_DURATION = 20.0
TARGET_DESTINATION_REACH_TIMEOUT = 10.0
MIN_TARGET_DISTANCE = 10.0
MAX_TARGET_DISTANCE = AREA_SIZE / 2.5
MAX_TARGET_GENERATION_ATTEMPTS = 20

# --- 동적 이벤트 스케줄 ---
DYNAMIC_EVENT_SCHEDULE = [
    (3.0, "ADD_OBSTACLE", {"count": 1, "size_range": ((0.5, 1.5), (1.0, 3.0), (0.5, 1.5))}),
    (5.0, "REMOVE_OBSTACLE", {"count": 1, "strategy": "random"}),
    (7.0, "ADD_OBSTACLE", {"count": 1, "size_range": ((1.0, 2.0), (1.5, 3.5), (1.0, 2.0))}),
    (9.0, "CHANGE_MAP_SEED", {"new_seed_offset": 100}),
    (11.0, "REMOVE_OBSTACLE", {"count": 1, "strategy": "random"}),
    (13.0, "ADD_OBSTACLE", {"count": 1, "size_range": ((2.0, 1.5), (2.0, 3.0), (1.5, 1.1))}),
    (15.0, "REMOVE_OBSTACLE", {"count": 1, "strategy": "random"}),
    (17.0, "ADD_OBSTACLE", {"count": 1, "size_range": ((2.5, 1.5), (2.5, 2.0), (0.5, 2.1))})
]

# --- 클러스터링 알고리즘 설정값 ---
DBSCAN_EPS = 1
DBSCAN_MIN_SAMPLES = 3
Y_COORD_SCALE_FACTOR = 0.2 # DBSCAN, HDBSCAN에 적용 가능
# KMEANS_N_CLUSTERS 와 GMM_N_COMPONENTS 는 evaluate_and_plot_algorithms 함수 내에서 동적으로 설정됨
GMM_COVARIANCE_TYPE = 'full'
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = None # None이면 min_cluster_size와 동일하게 설정됨
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0 # 0.0이면 HDBSCAN* 알고리즘 사용

# --- RANSAC 지면 분할 설정값 ---
RANSAC_DISTANCE_THRESHOLD = 0.5
RANSAC_N_POINTS = 3
RANSAC_NUM_ITERATIONS = 100
VISUALIZE_RANSAC_SEGMENTATION = False

# --- 성능 평가 설정값 ---
POSSIBLE_EVALUATION_METHODS = ["3D_IOU", "2D_CENTER_IN_GT_AREA", "2D_AREA_OVERLAP_RATIO"]
IOU_THRESHOLD = 0.5
AREA_OVERLAP_RATIO_THRESHOLD = 0.3 # 2D_AREA_OVERLAP_RATIO 에 사용될 임계값

# --- Ursina 애플리케이션 설정 ---
app = Ursina(
    title="Dynamic 3D LiDAR Simulator (A* Autonomous Driving)",
    borderless=False,
)

# --- 마우스 설정 ---
mouse.visible = True
mouse.locked = False

# --- 전역 변수 초기화 ---
scan_timer = 0.0
scan_history = []
obstacle_df = pd.DataFrame()
ground_info = {'size': AREA_SIZE, 'center_x': 0, 'center_z': 0}
movement_states = {'forward': False, 'backward': False, 'left': False, 'right': False}
rotation_states = {'left': False, 'right': False}
scene_obstacles_info = []
path_visualization_entities = []

# --- 자율 주행 및 시뮬레이션 상태 관련 전역 변수 ---
simulation_running = False
simulation_time_elapsed = 0.0
current_grid_map = None
current_path_waypoints = []
current_waypoint_idx = 0
target_destination_world = None
replanning_needed = False
next_dynamic_event_idx = 0
current_destination_timer = 0.0

# --- 그리드맵 클래스 (A* 알고리즘용) ---
class GridMap:
    def __init__(self, area_size, cell_size, obstacles_df, padding_cells=OBSTACLE_PADDING_CELLS):
        self.area_size = area_size
        self.cell_size = cell_size
        self.grid_dim = int(area_size / cell_size)
        self.grid = np.zeros((self.grid_dim, self.grid_dim), dtype=bool)
        self.origin_offset = area_size / 2.0
        self.padding_cells = padding_cells

        if obstacles_df is not None and not obstacles_df.empty:
            for _, obs in obstacles_df.iterrows():
                min_x_obs_orig = obs['center_x'] - obs['width'] / 2
                max_x_obs_orig = obs['center_x'] + obs['width'] / 2
                min_z_obs_orig = obs['center_z'] - obs['depth'] / 2
                max_z_obs_orig = obs['center_z'] + obs['depth'] / 2

                start_gx_orig, start_gz_orig = self.get_grid_coords(min_x_obs_orig, min_z_obs_orig)
                end_gx_orig, end_gz_orig = self.get_grid_coords(max_x_obs_orig, max_z_obs_orig)

                start_gx_padded = max(0, start_gx_orig - self.padding_cells)
                start_gz_padded = max(0, start_gz_orig - self.padding_cells)
                end_gx_padded = min(self.grid_dim -1, end_gx_orig + self.padding_cells)
                end_gz_padded = min(self.grid_dim -1, end_gz_orig + self.padding_cells)

                for gx in range(start_gx_padded, end_gx_padded + 1):
                    for gz in range(start_gz_padded, end_gz_padded + 1):
                        if 0 <= gx < self.grid_dim and 0 <= gz < self.grid_dim :
                             self.grid[gx, gz] = True

    def get_grid_coords(self, world_x, world_z):
        grid_x = int((world_x + self.origin_offset) / self.cell_size)
        grid_z = int((world_z + self.origin_offset) / self.cell_size)
        return grid_x, grid_z

    def get_world_coords(self, grid_x, grid_z):
        world_x = (grid_x * self.cell_size) - self.origin_offset + (self.cell_size / 2.0)
        world_z = (grid_z * self.cell_size) - self.origin_offset + (self.cell_size / 2.0)
        return world_x, world_z

    def is_walkable(self, grid_x, grid_z):
        if 0 <= grid_x < self.grid_dim and 0 <= grid_z < self.grid_dim:
            return not self.grid[grid_x, grid_z]
        return False

    def get_neighbors(self, grid_node_coords):
        gx, gz = grid_node_coords
        neighbors = []
        for dx in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dz == 0:
                    continue
                ng_x, ng_z = gx + dx, gz + dz
                if self.is_walkable(ng_x, ng_z):
                    cost = 1.414 if abs(dx) == 1 and abs(dz) == 1 else 1.0
                    neighbors.append(((ng_x, ng_z), cost))
        return neighbors

# --- A* 경로 탐색 함수 ---
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(grid_map, start_world_pos, goal_world_pos):
    if grid_map is None:
        print("A* Error: GridMap is not available.")
        return []

    start_node = grid_map.get_grid_coords(start_world_pos[0], start_world_pos[1])
    goal_node = grid_map.get_grid_coords(goal_world_pos[0], goal_world_pos[1])

    if not grid_map.is_walkable(start_node[0], start_node[1]):
        print(f"A* Warning: Start node {start_node} is not walkable.")
        return []
    if not grid_map.is_walkable(goal_node[0], goal_node[1]):
        print(f"A* Warning: Goal node {goal_node} is not walkable.")
        return []

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start_node, goal_node), 0, start_node))

    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: heuristic(start_node, goal_node)}

    path_found_count = 0

    while open_set:
        path_found_count +=1
        if path_found_count > grid_map.grid_dim * grid_map.grid_dim * 2 :
            print("A* search timed out.")
            return []

        current_f, current_g, current_node = heapq.heappop(open_set)

        if current_node == goal_node:
            path_grid_coords = []
            temp = current_node
            while temp in came_from:
                path_grid_coords.append(temp)
                temp = came_from[temp]
            path_grid_coords.append(start_node)
            path_grid_coords.reverse()

            world_path = [Vec3(grid_map.get_world_coords(gc[0], gc[1])[0], AGENT_HEIGHT, grid_map.get_world_coords(gc[0], gc[1])[1]) for gc in path_grid_coords]
            return world_path

        for neighbor_node, cost_to_neighbor in grid_map.get_neighbors(current_node):
            tentative_g_score = current_g + cost_to_neighbor
            if tentative_g_score < g_score.get(neighbor_node, float('inf')):
                came_from[neighbor_node] = current_node
                g_score[neighbor_node] = tentative_g_score
                new_f_score = tentative_g_score + heuristic(neighbor_node, goal_node)
                f_score[neighbor_node] = new_f_score

                heapq.heappush(open_set, (new_f_score, tentative_g_score, neighbor_node))

    print("A* path not found.")
    return []

# --- 경로 시각화 함수 ---
def visualize_path(path_waypoints):
    global path_visualization_entities
    for entity in path_visualization_entities:
        destroy(entity)
    path_visualization_entities.clear()

    if not path_waypoints:
        return

    if len(path_waypoints) > 1:
        path_vis_y_offset = AGENT_HEIGHT + 0.1
        adjusted_waypoints = [Vec3(wp.x, path_vis_y_offset, wp.z) for wp in path_waypoints]
        path_mesh = Mesh(vertices=adjusted_waypoints, mode='line', thickness=4)
        path_entity = Entity(model=path_mesh, color=color.yellow, alpha=0.9)
        path_visualization_entities.append(path_entity)

# --- 환경 요소 생성 함수 ---
def create_initial_environment():
    global ground, sky
    ground = Entity(model='plane', scale=(AREA_SIZE,1,AREA_SIZE), color=color.light_gray, texture='white_cube', texture_scale=(AREA_SIZE/2,AREA_SIZE/2), collider='box', tag="ground")
    sky = Sky()

def clear_obstacles():
    global scene_obstacles_info, obstacle_df
    for obs_info in scene_obstacles_info:
        if obs_info['entity'] and obs_info['entity'].enabled:
            destroy(obs_info['entity'])
    scene_obstacles_info = []
    obstacle_df = pd.DataFrame(columns=obstacle_df.columns if not obstacle_df.empty else ['name','center_x','center_y','center_z','width','height','depth','rotation_y'])

def generate_obstacles(seed_value, num_to_generate=NUM_OBSTACLES):
    global scene_obstacles_info, obstacle_df, current_obstacle_seed, max_placement_attempts_per_obstacle, min_obstacle_distance_from_spawn
    global current_grid_map, replanning_needed

    current_obstacle_seed = seed_value
    random.seed(current_obstacle_seed)
    print(f"Generating obstacles... Seed: {current_obstacle_seed}, Count: {num_to_generate}")

    temp_obstacles_list_for_df = []
    temp_placed_aabbs = [info['aabb'] for info in scene_obstacles_info if info['entity'] and info['entity'].enabled]

    for i in range(num_to_generate):
        placed_successfully = False
        for attempt in range(max_placement_attempts_per_obstacle):
            pos_x_cand = random.uniform(-AREA_SIZE / 2.8, AREA_SIZE / 2.8)
            pos_z_cand = random.uniform(-AREA_SIZE / 2.8, AREA_SIZE / 2.8)
            scale_x_cand = random.uniform(0.5, 3.0); scale_y_cand = random.uniform(1.0, 5.0); scale_z_cand = random.uniform(0.5, 3.0)
            pos_y_cand = scale_y_cand / 2
            cand_min_x = pos_x_cand - scale_x_cand/2; cand_max_x = pos_x_cand + scale_x_cand/2
            cand_min_y = pos_y_cand - scale_y_cand/2; cand_max_y = pos_y_cand + scale_y_cand/2
            cand_min_z = pos_z_cand - scale_z_cand/2; cand_max_z = pos_z_cand + scale_z_cand/2

            agent_current_pos = agent.position if agent else Vec3(0,0,0)
            dist_from_agent_spawn = math.sqrt((pos_x_cand - agent_current_pos.x)**2 + (pos_z_cand - agent_current_pos.z)**2)
            if dist_from_agent_spawn < min_obstacle_distance_from_spawn + max(scale_x_cand/2, scale_z_cand/2):
                continue

            overlap_found = False
            for placed_aabb in temp_placed_aabbs:
                x_overlap = (cand_max_x > placed_aabb['min_x'] and cand_min_x < placed_aabb['max_x'])
                y_overlap = (cand_max_y > placed_aabb['min_y'] and cand_min_y < placed_aabb['max_y'])
                z_overlap = (cand_max_z > placed_aabb['min_z'] and cand_min_z < placed_aabb['max_z'])
                if x_overlap and y_overlap and z_overlap: overlap_found = True; break

            if not overlap_found:
                obstacle_color = color.random_color()
                obs_name = f"obstacle_s{current_obstacle_seed}_{len(scene_obstacles_info)}_{i}"
                obs_entity = Entity(model='cube',position=(pos_x_cand,pos_y_cand,pos_z_cand),scale=(scale_x_cand,scale_y_cand,scale_z_cand),color=obstacle_color,collider='box',tag=OBSTACLE_TAG,name=obs_name)

                df_data = {'name':obs_name,'center_x':pos_x_cand,'center_y':pos_y_cand,'center_z':pos_z_cand,'width':scale_x_cand,'height':scale_y_cand,'depth':scale_z_cand,'rotation_y':0}
                aabb_data = {'min_x':cand_min_x,'max_x':cand_max_x,'min_y':cand_min_y,'max_y':cand_max_y,'min_z':cand_min_z,'max_z':cand_max_z}

                scene_obstacles_info.append({'entity': obs_entity, 'df_data': df_data, 'aabb': aabb_data})
                temp_obstacles_list_for_df.append(df_data)
                temp_placed_aabbs.append(aabb_data)
                placed_successfully = True; break
        if not placed_successfully: print(f"Warning: Obstacle index {i} could not be placed without overlap after {max_placement_attempts_per_obstacle} attempts.")

    if temp_obstacles_list_for_df:
        obstacle_df = pd.concat([obstacle_df, pd.DataFrame(temp_obstacles_list_for_df)], ignore_index=True)
    print(f"Obstacle DataFrame updated ({len(obstacle_df)} total).")

    current_grid_map = GridMap(AREA_SIZE, GRID_CELL_SIZE, obstacle_df, padding_cells=OBSTACLE_PADDING_CELLS)
    replanning_needed = True


def add_dynamic_obstacle(count=1, size_range=((0.5, 2.0), (1.0, 4.0), (0.5, 2.0))):
    global scene_obstacles_info, obstacle_df, max_placement_attempts_per_obstacle
    global current_grid_map, replanning_needed

    print(f"Attempting to add {count} dynamic obstacle(s)...")

    temp_obstacles_list_for_df = []
    temp_placed_aabbs = [info['aabb'] for info in scene_obstacles_info if info['entity'] and info['entity'].enabled]

    added_count = 0
    for i in range(count):
        placed_successfully = False
        for attempt in range(max_placement_attempts_per_obstacle):
            angle = random.uniform(0, 2 * math.pi)
            distance_from_agent = random.uniform(LIDAR_RANGE * 0.3, LIDAR_RANGE * 1.2)
            offset_x = math.sin(angle) * distance_from_agent
            offset_z = math.cos(angle) * distance_from_agent
            pos_x_cand = agent.x + offset_x
            pos_z_cand = agent.z + offset_z

            if not (-AREA_SIZE/2.2 < pos_x_cand < AREA_SIZE/2.2 and -AREA_SIZE/2.2 < pos_z_cand < AREA_SIZE/2.2):
                continue

            scale_x_cand = random.uniform(size_range[0][0], size_range[0][1])
            scale_y_cand = random.uniform(size_range[1][0], size_range[1][1])
            scale_z_cand = random.uniform(size_range[2][0], size_range[2][1])
            pos_y_cand = scale_y_cand / 2

            cand_min_x = pos_x_cand - scale_x_cand/2; cand_max_x = pos_x_cand + scale_x_cand/2
            cand_min_y = pos_y_cand - scale_y_cand/2; cand_max_y = pos_y_cand + scale_y_cand/2
            cand_min_z = pos_z_cand - scale_z_cand/2; cand_max_z = pos_z_cand + scale_z_cand/2

            agent_half_scale_x = agent.scale_x / 2
            agent_half_scale_z = agent.scale_z / 2 if hasattr(agent, 'scale_z') else agent_half_scale_x
            agent_min_x = agent.x - agent_half_scale_x; agent_max_x = agent.x + agent_half_scale_x
            agent_min_z = agent.z - agent_half_scale_z; agent_max_z = agent.z + agent_half_scale_z

            agent_overlap = (cand_max_x > agent_min_x and cand_min_x < agent_max_x and \
                             cand_max_z > agent_min_z and cand_min_z < agent_max_z)
            if agent_overlap: continue

            overlap_with_existing = False
            for placed_aabb in temp_placed_aabbs:
                x_overlap = (cand_max_x > placed_aabb['min_x'] and cand_min_x < placed_aabb['max_x'])
                y_overlap = (cand_max_y > placed_aabb['min_y'] and cand_min_y < placed_aabb['max_y'])
                z_overlap = (cand_max_z > placed_aabb['min_z'] and cand_min_z < placed_aabb['max_z'])
                if x_overlap and y_overlap and z_overlap: overlap_with_existing = True; break

            if not overlap_with_existing:
                obstacle_color = color.lime
                obs_name = f"obstacle_dynamic_{len(scene_obstacles_info)}_{i}"
                obs_entity = Entity(model='cube',position=(pos_x_cand,pos_y_cand,pos_z_cand),scale=(scale_x_cand,scale_y_cand,scale_z_cand),color=obstacle_color,collider='box',tag=OBSTACLE_TAG,name=obs_name)

                df_data = {'name':obs_name,'center_x':pos_x_cand,'center_y':pos_y_cand,'center_z':pos_z_cand,'width':scale_x_cand,'height':scale_y_cand,'depth':scale_z_cand,'rotation_y':0}
                aabb_data = {'min_x':cand_min_x,'max_x':cand_max_x,'min_y':cand_min_y,'max_y':cand_max_y,'min_z':cand_min_z,'max_z':cand_max_z}

                scene_obstacles_info.append({'entity': obs_entity, 'df_data': df_data, 'aabb': aabb_data})
                temp_obstacles_list_for_df.append(df_data)
                temp_placed_aabbs.append(aabb_data)
                placed_successfully = True; added_count +=1; break
        if not placed_successfully: print(f"Warning: Dynamic obstacle {i} failed to place.")

    if temp_obstacles_list_for_df:
        obstacle_df = pd.concat([obstacle_df, pd.DataFrame(temp_obstacles_list_for_df)], ignore_index=True)
        print(f"{added_count} new dynamic obstacle(s) created. Total obstacles: {len(obstacle_df)}")
        current_grid_map = GridMap(AREA_SIZE, GRID_CELL_SIZE, obstacle_df, padding_cells=OBSTACLE_PADDING_CELLS)
        replanning_needed = True


def remove_dynamic_obstacle(count=1, strategy="closest"):
    global scene_obstacles_info, obstacle_df
    global current_grid_map, replanning_needed

    if not scene_obstacles_info:
        print("No obstacles to remove.")
        return

    print(f"Attempting to remove {count} dynamic obstacle(s) (strategy: {strategy})...")
    removed_count = 0
    for _ in range(count):
        enabled_obstacles_with_indices = [(i, obs_info) for i, obs_info in enumerate(scene_obstacles_info) if obs_info['entity'] and obs_info['entity'].enabled]
        if not enabled_obstacles_with_indices:
            print("No enabled obstacles left to remove.")
            break

        obs_to_remove_original_idx = -1
        obs_to_remove_info = None

        if strategy == "closest":
            min_dist_sq = float('inf')
            current_closest_original_idx = -1
            for original_idx, obs_info_item in enabled_obstacles_with_indices:
                dist_sq = distance_squared(agent.world_position, obs_info_item['entity'].world_position)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    obs_to_remove_original_idx = original_idx
                    obs_to_remove_info = obs_info_item

        elif strategy == "random":
            if enabled_obstacles_with_indices:
                chosen_original_idx, chosen_obs_info = random.choice(enabled_obstacles_with_indices)
                obs_to_remove_original_idx = chosen_original_idx
                obs_to_remove_info = chosen_obs_info

        if obs_to_remove_original_idx != -1 and obs_to_remove_info:
            print(f"Removing obstacle '{obs_to_remove_info['df_data']['name']}'...")
            destroy(obs_to_remove_info['entity'])

            obstacle_df = obstacle_df[obstacle_df['name'] != obs_to_remove_info['df_data']['name']].reset_index(drop=True)
            scene_obstacles_info = [item for item in scene_obstacles_info if item['df_data']['name'] != obs_to_remove_info['df_data']['name']]
            removed_count += 1
        else:
            print("Could not find a suitable obstacle to remove (or already removed).")
            break

    if removed_count > 0:
        print(f"{removed_count} obstacle(s) removed. Remaining obstacles: {len(scene_obstacles_info)}")
        current_grid_map = GridMap(AREA_SIZE, GRID_CELL_SIZE, obstacle_df, padding_cells=OBSTACLE_PADDING_CELLS)
        replanning_needed = True

# --- 에이전트 및 카메라 설정 ---
agent = Entity(model='sphere', color=color.blue, position=(0,AGENT_HEIGHT,0), collider='sphere', scale=1)
camera.parent = agent; camera.position = (0,10,-8); camera.rotation_x = 45; camera.rotation_y = 0; camera.fov = 75

# --- 유틸리티 함수 ---
def normalize_angle(rad):
    while rad > math.pi: rad -= 2*math.pi
    while rad < -math.pi: rad += 2*math.pi
    return rad

# --- 3D 라이다 시각화 함수 ---
lidar_lines = []
def update_lidar_visualization():
    global lidar_lines
    for line in lidar_lines: destroy(line)
    lidar_lines.clear()
    agent_pos=agent.world_position; agent_rot_y_deg=agent.world_rotation_y
    lidar_origin_world = agent_pos + Vec3(0,LIDAR_VIS_HEIGHT,0)
    h_angle_step=LIDAR_FOV/(NUM_LIDAR_RAYS-1) if NUM_LIDAR_RAYS>1 else 0
    v_angle_step=LIDAR_VERTICAL_FOV/(NUM_LIDAR_CHANNELS-1) if NUM_LIDAR_CHANNELS>1 else 0
    v_angle_start=-LIDAR_VERTICAL_FOV/2
    for ch in range(NUM_LIDAR_CHANNELS):
        v_angle_deg=v_angle_start+ch*v_angle_step; v_angle_rad=math.radians(v_angle_deg)
        cos_v=math.cos(v_angle_rad); sin_v=math.sin(v_angle_rad)
        h_angle_start_world_deg=agent_rot_y_deg-LIDAR_FOV/2
        for i in range(NUM_LIDAR_RAYS):
            h_angle_world_deg=h_angle_start_world_deg+i*h_angle_step
            h_angle_world_rad=math.radians(h_angle_world_deg)
            cos_h=math.cos(h_angle_world_rad); sin_h=math.sin(h_angle_world_rad)
            direction=Vec3(cos_v*sin_h,sin_v,cos_v*cos_h).normalized()
            hit_info=raycast(lidar_origin_world,direction,LIDAR_RANGE,ignore=[agent, ground],debug=False,traverse_target=scene)
            if hit_info.hit:
                end_point=hit_info.world_point
                is_obstacle=hasattr(hit_info.entity,'tag') and hit_info.entity.tag==OBSTACLE_TAG
                is_ground_entity = hit_info.entity.tag == "ground" if hasattr(hit_info.entity, 'tag') else False
                line_color=color.red if is_obstacle else (color.dark_gray if is_ground_entity else color.orange)
            else: end_point=lidar_origin_world+direction*LIDAR_RANGE; line_color=LIDAR_COLOR
            if distance(lidar_origin_world,end_point)>0.01: lidar_lines.append(Entity(model=Mesh(vertices=[lidar_origin_world,end_point],mode='line',thickness=1),color=line_color,alpha=0.7))

# --- 3D 라이다 스캔 및 상대 좌표 데이터 생성 함수 (노이즈 추가 기능 포함) ---
def generate_relative_lidar_map_3d_at_pose(scan_agent_world_pos, scan_agent_world_rotation_y_deg):
    relative_points_3d=[]
    agent_pos_world = scan_agent_world_pos
    agent_rot_y_rad = math.radians(scan_agent_world_rotation_y_deg)

    scan_origin_world=agent_pos_world+Vec3(0,LIDAR_VIS_HEIGHT,0)
    h_angle_step=LIDAR_FOV/(NUM_LIDAR_RAYS-1) if NUM_LIDAR_RAYS>1 else 0
    v_angle_step=LIDAR_VERTICAL_FOV/(NUM_LIDAR_CHANNELS-1) if NUM_LIDAR_CHANNELS>1 else 0
    v_angle_start=-LIDAR_VERTICAL_FOV/2
    for ch in range(NUM_LIDAR_CHANNELS):
        v_angle_deg=v_angle_start+ch*v_angle_step; v_angle_rad=math.radians(v_angle_deg)
        cos_v=math.cos(v_angle_rad); sin_v=math.sin(v_angle_rad)
        h_angle_start_world_deg = scan_agent_world_rotation_y_deg - LIDAR_FOV/2

        for i in range(NUM_LIDAR_RAYS):
            h_angle_world_deg=h_angle_start_world_deg+i*h_angle_step
            h_angle_world_rad=math.radians(h_angle_world_deg)
            cos_h=math.cos(h_angle_world_rad); sin_h=math.sin(h_angle_world_rad)
            direction=Vec3(cos_v*sin_h,sin_v,cos_v*cos_h).normalized()
            hit_info=raycast(scan_origin_world,direction,LIDAR_RANGE,ignore=[agent,],debug=False,traverse_target=scene)
            if hit_info.hit:
                hit_point_world=hit_info.world_point
                
                # --- 노이즈 추가 로직 ---
                if LIDAR_NOISE_ENABLED:
                    noise_x = random.uniform(-LIDAR_NOISE_LEVEL, LIDAR_NOISE_LEVEL)
                    noise_y = random.uniform(-LIDAR_NOISE_LEVEL, LIDAR_NOISE_LEVEL)
                    noise_z = random.uniform(-LIDAR_NOISE_LEVEL, LIDAR_NOISE_LEVEL)
                    hit_point_world = Vec3(hit_point_world.x + noise_x,
                                           hit_point_world.y + noise_y,
                                           hit_point_world.z + noise_z)
                # --- 노이즈 추가 로직 끝 ---

                world_vec=hit_point_world-agent_pos_world

                cos_a=math.cos(-agent_rot_y_rad); sin_a=math.sin(-agent_rot_y_rad)
                relative_x=world_vec.x*cos_a-world_vec.z*sin_a; relative_y=world_vec.y; relative_z=world_vec.x*sin_a+world_vec.z*cos_a
                relative_points_3d.append([relative_x,relative_y,relative_z])
    return np.array(relative_points_3d) if relative_points_3d else np.empty((0,3))

# --- 누적된 스캔 데이터로부터 전역 3D 맵 데이터 생성 함수 ---
def generate_map_data_3d(history):
    global_map_points_list=[]; agent_trajectory_x,agent_trajectory_z=[],[]
    if not history: return np.empty((0,3)),[],[]
    for scan_record in history:
        pose=scan_record['pose']; relative_points=np.array(scan_record['relative_points'])
        if relative_points.shape[0]==0: continue
        # pose는 (월드_X, 월드_Z, 월드_Yaw_라디안) 형태
        scan_pos_x,scan_pos_z,scan_rot_rad=pose[0], pose[1], pose[2]
        agent_pos_world_at_scan=Vec3(scan_pos_x,AGENT_HEIGHT,scan_pos_z)
        agent_trajectory_x.append(scan_pos_x); agent_trajectory_z.append(scan_pos_z)
        cos_a,sin_a=math.cos(scan_rot_rad),math.sin(scan_rot_rad)
        rot_matrix_2d=np.array([[cos_a,-sin_a],[sin_a,cos_a]])
        relative_xz=relative_points[:,[0,2]]
        rotated_xz=np.dot(relative_xz,rot_matrix_2d.T)
        world_y = relative_points[:,1] + agent_pos_world_at_scan.y # 스캔 당시 에이전트 Y좌표 기준 상대 Y를 더함

        world_x=rotated_xz[:,0]+scan_pos_x;
        world_z=rotated_xz[:,1]+scan_pos_z
        world_points=np.stack((world_x,world_y,world_z),axis=-1)
        global_map_points_list.append(world_points)
    map_points_np=np.concatenate(global_map_points_list,axis=0) if global_map_points_list else np.empty((0,3))
    return map_points_np,agent_trajectory_x,agent_trajectory_z

# --- 포인트 클라우드 전처리 함수: RANSAC 지면 분할 ---
def segment_ground_ransac(points_np, distance_threshold=0.2, ransac_n=3, num_iterations=100, visualize_segmentation=False):
    if points_np.shape[0]<ransac_n: return points_np,np.empty((0,3))
    pcd=o3d.geometry.PointCloud(); pcd.points=o3d.utility.Vector3dVector(points_np)
    try:
        plane_model,inliers=pcd.segment_plane(distance_threshold,ransac_n,num_iterations)
        ground_pcd=pcd.select_by_index(inliers); non_ground_pcd=pcd.select_by_index(inliers,invert=True)
        if visualize_segmentation:
            ground_pcd.paint_uniform_color([0.5,0.5,0.5]); non_ground_pcd.paint_uniform_color([1,0,0])
            o3d.visualization.draw_geometries([ground_pcd,non_ground_pcd],"RANSAC Segmentation",800,600)
        return np.asarray(non_ground_pcd.points),np.asarray(ground_pcd.points)
    except Exception as e: print(f"RANSAC error: {e}"); return points_np,np.empty((0,3))

# --- 3D 직육면체(Bounding Box) 그리기 헬퍼 함수 ---
def draw_cuboid(ax,min_coords,max_coords,color='r',alpha=0.1,linewidth=1,label=None):
    min_x,min_y_h,min_z_d=min_coords; max_x,max_y_h,max_z_d=max_coords
    v=np.array([(min_x,min_z_d,min_y_h),(max_x,min_z_d,min_y_h),(max_x,max_z_d,min_y_h),(min_x,max_z_d,min_y_h),
                (min_x,min_z_d,max_y_h),(max_x,min_z_d,max_y_h),(max_x,max_z_d,max_y_h),(min_x,max_z_d,max_y_h)])
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    plotted=False
    for i,edge in enumerate(edges):
        xs,zs,ys=v[edge,0],v[edge,1],v[edge,2]
        lbl=label if not plotted else None
        ax.plot(xs,zs,ys,color=color,alpha=alpha*2,linewidth=linewidth,label=lbl)
        if lbl: plotted=True

# --- 3D IoU 계산 함수 ---
def calculate_3d_iou(box1_min_xyz,box1_max_xyz,box2_min_xyz,box2_max_xyz):
    inter_min_x=max(box1_min_xyz[0],box2_min_xyz[0]); inter_min_y=max(box1_min_xyz[1],box2_min_xyz[1])
    inter_min_z=max(box1_min_xyz[2],box2_min_xyz[2]); inter_max_x=min(box1_max_xyz[0],box2_max_xyz[0])
    inter_max_y=min(box1_max_xyz[1],box2_max_xyz[1]); inter_max_z=min(box1_max_xyz[2],box2_max_xyz[2])
    inter_w=max(0,inter_max_x-inter_min_x); inter_h=max(0,inter_max_y-inter_min_y)
    inter_d=max(0,inter_max_z-inter_min_z); intersection_volume=inter_w*inter_h*inter_d
    vol1=(box1_max_xyz[0]-box1_min_xyz[0])*(box1_max_xyz[1]-box1_min_xyz[1])*(box1_max_xyz[2]-box1_min_xyz[2])
    vol2=(box2_max_xyz[0]-box2_min_xyz[0])*(box2_max_xyz[1]-box2_min_xyz[1])*(box2_max_xyz[2]-box2_min_xyz[2])
    union_volume=vol1+vol2-intersection_volume
    return intersection_volume/union_volume if union_volume > 0 else 0.0

# --- XZ 평면상 겹침 비율 계산 함수 ---
def calculate_2d_xz_overlap_ratio(pred_min_xz, pred_max_xz, gt_min_xz, gt_max_xz):
    inter_min_x = max(pred_min_xz[0], gt_min_xz[0]); inter_min_z = max(pred_min_xz[1], gt_min_xz[1])
    inter_max_x = min(pred_max_xz[0], gt_max_xz[0]); inter_max_z = min(pred_max_xz[1], gt_max_xz[1])
    inter_w = max(0, inter_max_x - inter_min_x); inter_d = max(0, inter_max_z - inter_min_z)
    intersection_area = inter_w * inter_d
    gt_area = (gt_max_xz[0] - gt_min_xz[0]) * (gt_max_xz[1] - gt_min_xz[1])
    return intersection_area / gt_area if gt_area > 0 else 0.0


# --- 통합 맵 플로팅 및 모든 알고리즘 성능 평가 함수 ---
def evaluate_and_plot_algorithms(history, current_gt_obstacle_df, gnd_info, show_plots=True):
    print(f"Generating consolidated map and evaluating all algorithms...")
    if not history: print("No scan history to plot."); return None
    if current_gt_obstacle_df is None or current_gt_obstacle_df.empty : print("No Ground Truth obstacle data."); return None

    all_algorithms_results_summary_for_all_eval_methods = []

    try:
        map_points_np_3d, agent_traj_x, agent_traj_z = generate_map_data_3d(history)
        print(f"Total global 3D points generated: {len(map_points_np_3d)}")
        if len(map_points_np_3d) == 0 and not agent_traj_x:
            print("No points and no trajectory data generated."); return None

        relevant_gt_indices = set()
        if not current_gt_obstacle_df.empty and history:
            for gt_idx_val, gt_row in current_gt_obstacle_df.iterrows():
                gt_center_world = Vec3(gt_row['center_x'], gt_row['center_y'], gt_row['center_z'])
                gt_half_extents = Vec3(gt_row['width']/2, gt_row['height']/2, gt_row['depth']/2)

                for scan_record in history:
                    agent_scan_pos_world = Vec3(scan_record['pose'][0], AGENT_HEIGHT + LIDAR_VIS_HEIGHT, scan_record['pose'][1])
                    dist_to_gt_center = distance(agent_scan_pos_world, gt_center_world)
                    gt_diagonal_half = distance(Vec3(0,0,0), gt_half_extents)

                    if dist_to_gt_center - gt_diagonal_half < LIDAR_RANGE :
                         relevant_gt_indices.add(gt_idx_val); break
        relevant_gt_obstacle_df = current_gt_obstacle_df.loc[list(relevant_gt_indices)].copy() if relevant_gt_indices else pd.DataFrame(columns=current_gt_obstacle_df.columns)

        non_ground_points_orig = np.empty((0,3))
        ground_points_ransac = np.empty((0,3))

        if len(map_points_np_3d) > 0:
            non_ground_points_orig_temp, ground_points_ransac_temp = segment_ground_ransac(map_points_np_3d,RANSAC_DISTANCE_THRESHOLD,RANSAC_N_POINTS,RANSAC_NUM_ITERATIONS,VISUALIZE_RANSAC_SEGMENTATION)
            if non_ground_points_orig_temp.ndim == 2 and non_ground_points_orig_temp.shape[0] > 0 :
                non_ground_points_orig = non_ground_points_orig_temp
            if ground_points_ransac_temp.ndim == 2 and ground_points_ransac_temp.shape[0] > 0:
                 ground_points_ransac = ground_points_ransac_temp

        if len(non_ground_points_orig) == 0:
            print("No non-ground points after RANSAC. Cannot proceed with evaluation.")
            return None

        num_relevant_gt_obstacles = len(relevant_gt_obstacle_df)
        print(f"Number of relevant GT obstacles in LiDAR range for clustering: {num_relevant_gt_obstacles}")

        if num_relevant_gt_obstacles == 0:
            dynamic_k_for_kmeans = 1
            dynamic_n_for_gmm = 1
        elif num_relevant_gt_obstacles == 1:
            dynamic_k_for_kmeans = 1
            dynamic_n_for_gmm = 1
        else:
            dynamic_k_for_kmeans = num_relevant_gt_obstacles
            dynamic_n_for_gmm = num_relevant_gt_obstacles

        MAX_CLUSTERS_FOR_KM_GMM = NUM_OBSTACLES #15
        dynamic_k_for_kmeans = min(dynamic_k_for_kmeans, MAX_CLUSTERS_FOR_KM_GMM)
        dynamic_n_for_gmm = min(dynamic_n_for_gmm, MAX_CLUSTERS_FOR_KM_GMM)
        print(f"Dynamic K for K-Means: {dynamic_k_for_kmeans}, Dynamic N for GMM: {dynamic_n_for_gmm}")


        algorithms_to_run = [
            {"name": "DBSCAN", "model_params": {"eps": DBSCAN_EPS, "min_samples": DBSCAN_MIN_SAMPLES}, "y_scale": Y_COORD_SCALE_FACTOR},
            {"name": "HDBSCAN", "model_params": {"min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE, "min_samples": HDBSCAN_MIN_SAMPLES, "cluster_selection_epsilon": HDBSCAN_CLUSTER_SELECTION_EPSILON, "allow_single_cluster": True}, "y_scale": Y_COORD_SCALE_FACTOR},
            {"name": "KMEANS", "model_params": {"n_clusters": dynamic_k_for_kmeans, "random_state": current_obstacle_seed, "n_init": 'auto'}},
            {"name": "GMM", "model_params": {"n_components": dynamic_n_for_gmm, "covariance_type": GMM_COVARIANCE_TYPE, "random_state": current_obstacle_seed}}
        ]

        # gt_boxes_info_for_eval는 모든 평가 방법에 대해 동일하므로 한 번만 생성
        gt_boxes_info_for_eval = []
        for idx, row in relevant_gt_obstacle_df.iterrows():
            min_x_gt=row['center_x']-row['width']/2; max_x_gt=row['center_x']+row['width']/2
            min_y_gt=row['center_y']-row['height']/2; max_y_gt=row['center_y']+row['height']/2
            min_z_gt=row['center_z']-row['depth']/2; max_z_gt=row['center_z']+row['depth']/2
            gt_boxes_info_for_eval.append({
                'min_coords_3d':(min_x_gt,min_y_gt,min_z_gt),
                'max_coords_3d':(max_x_gt,max_y_gt,max_z_gt),
                'min_coords_2d_xz':(min_x_gt,min_z_gt),
                'max_coords_2d_xz':(max_x_gt,max_z_gt),
                'width': row['width'], 'depth': row['depth'], 'center_x': row['center_x'], 'center_z': row['center_z'],
                'is_detected': False, # 각 평가 방법 루프 내에서 초기화될 것임
                'original_gt_idx':idx
            })


        for current_eval_method in POSSIBLE_EVALUATION_METHODS:
            print(f"\n===== Evaluating with Method: {current_eval_method} =====")
            for algo_config in algorithms_to_run:
                algorithm_name = algo_config["name"]
                model_params = algo_config["model_params"]
                y_scale_factor_algo = algo_config.get("y_scale", 1.0)

                print(f"\n--- Evaluating: [{algorithm_name}] using [{current_eval_method}] ---")

                points_for_algo_input = non_ground_points_orig.copy()
                if y_scale_factor_algo != 1.0 and algorithm_name in ["DBSCAN", "HDBSCAN"]: # Y 스케일링은 DBSCAN, HDBSCAN에만 적용
                    points_for_algo_input[:, 1] *= y_scale_factor_algo

                should_generate_plot_for_this_iteration = show_plots and (current_eval_method == "2D_AREA_OVERLAP_RATIO")
                fig_algo = None
                ax1, ax2, ax3 = None, None, None 

                if should_generate_plot_for_this_iteration:
                    fig_algo=plt.figure(figsize=(24,8.5)) 
                    fig_algo.suptitle(f'3D LiDAR Mapping - Algorithm: {algorithm_name} & Performance (Eval: {current_eval_method})',fontsize=16)

                    ax1=fig_algo.add_subplot(131,projection='3d')
                    if len(map_points_np_3d)>0: ax1.scatter(map_points_np_3d[:,0],map_points_np_3d[:,2],map_points_np_3d[:,1],s=1,c=map_points_np_3d[:,1],cmap='Greys',alpha=0.1,label='Original')
                    if len(ground_points_ransac)>0: ax1.scatter(ground_points_ransac[:,0],ground_points_ransac[:,2],ground_points_ransac[:,1],s=1,c='lightgray',alpha=0.2,label='Ground (RANSAC)')
                    if len(non_ground_points_orig)>0: ax1.scatter(non_ground_points_orig[:,0],non_ground_points_orig[:,2],non_ground_points_orig[:,1],s=2,c=non_ground_points_orig[:,1],cmap='viridis',alpha=0.6,label='Non-Ground (Original)')
                    if agent_traj_x:
                        ax1.plot(agent_traj_x,agent_traj_z,zs=0,zdir='z',marker='.',linestyle='-',color='blue',label='Trajectory')
                        ax1.scatter(agent_traj_x[-1],agent_traj_z[-1],zs=0,zdir='z',s=60,c='magenta',marker='*',label='Last Pose')
                    ax1.set_title("Point Cloud (RANSAC Processed)"); ax1.set_xlabel("X"); ax1.set_ylabel("Z (Depth)"); ax1.set_zlabel("Y (Height)")
                    limit=AREA_SIZE/1.8; ax1.set_xlim(-limit,limit); ax1.set_ylim(-limit,limit); ax1.set_zlim(0,10)
                    ax1.view_init(elev=30.,azim=-60); ax1.legend(fontsize='small',markerscale=2)

                    ax2=fig_algo.add_subplot(132)
                    ax3=fig_algo.add_subplot(133,projection='3d')


                detected_obstacle_count=0; predicted_boxes_info=[]
                labels = np.array([])
                clustering_time_sec = 0.0

                min_samples_for_algo = 1
                if algorithm_name == "DBSCAN": min_samples_for_algo = model_params["min_samples"]
                elif algorithm_name == "HDBSCAN": min_samples_for_algo = model_params["min_cluster_size"] 
                elif algorithm_name == "KMEANS": min_samples_for_algo = max(1, model_params["n_clusters"])
                elif algorithm_name == "GMM": min_samples_for_algo = max(1, model_params["n_components"])


                if len(points_for_algo_input) >= min_samples_for_algo:
                    try:
                        current_algo_model_params = model_params.copy()
                        if algorithm_name == "KMEANS": current_algo_model_params["n_clusters"] = min(min_samples_for_algo, len(points_for_algo_input)) 
                        elif algorithm_name == "GMM": current_algo_model_params["n_components"] = min(min_samples_for_algo, len(points_for_algo_input)) 

                        # --- 클러스터링 시간 측정 시작 ---
                        time_start = time.perf_counter()
                        if algorithm_name == "DBSCAN": model = DBSCAN(**current_algo_model_params); labels = model.fit_predict(points_for_algo_input)
                        elif algorithm_name == "HDBSCAN": model = hdbscan.HDBSCAN(**current_algo_model_params); labels = model.fit_predict(points_for_algo_input) 
                        elif algorithm_name == "KMEANS": model = KMeans(**current_algo_model_params); labels = model.fit_predict(points_for_algo_input)
                        elif algorithm_name == "GMM": model = GaussianMixture(**current_algo_model_params); labels = model.fit_predict(points_for_algo_input)
                        clustering_time_sec = time.perf_counter() - time_start
                        print(f"  {algorithm_name} clustering took: {clustering_time_sec:.4f} sec")
                        # --- 클러스터링 시간 측정 끝 ---


                        if labels.size > 0 and len(non_ground_points_orig) == len(labels):
                            unique_labels = set(labels)
                            num_clusters = len(unique_labels) - (1 if -1 in labels and algorithm_name in ["DBSCAN", "HDBSCAN"] else 0)
                            if algorithm_name not in ["DBSCAN", "HDBSCAN"]: num_clusters = len(unique_labels)

                            if should_generate_plot_for_this_iteration:
                                colors_map = plt.cm.get_cmap('Spectral', len(unique_labels) if len(unique_labels) > 0 else 1)
                            cluster_label_printed = False
                            for k_idx, k_label in enumerate(unique_labels):
                                if algorithm_name in ["DBSCAN", "HDBSCAN"] and k_label == -1: continue
                                cluster_mask = (labels == k_label)
                                original_cluster_points = non_ground_points_orig[cluster_mask]
                                if len(original_cluster_points) > 0:
                                    min_c = np.min(original_cluster_points, axis=0); max_c = np.max(original_cluster_points, axis=0)
                                    if (max_c[0]-min_c[0]) < 0.1 or (max_c[2]-min_c[2]) < 0.1 or (max_c[1]-min_c[1]) < 0.1 : continue

                                    predicted_boxes_info.append({'min_coords': tuple(min_c), 'max_coords': tuple(max_c), 'is_tp': False, 'cluster_label': k_label})
                                    if should_generate_plot_for_this_iteration and ax3:
                                        lbl = f'{algorithm_name} BBox' if not cluster_label_printed else None
                                        draw_cuboid(ax3, min_c, max_c, color=colors_map(k_idx/len(unique_labels) if len(unique_labels)>0 else 0.5), alpha=0.15, linewidth=1.5, label=lbl)
                                        if lbl: cluster_label_printed = True
                                    detected_obstacle_count += 1
                    except Exception as e:
                        print(f"  {algorithm_name} Clustering/Plotting error: {e}")
                        if should_generate_plot_for_this_iteration and ax3: ax3.text(0,0,0,f'{algorithm_name} Error',c='red')
                else:
                    print(f"  Not enough points for {algorithm_name} (or k/min_cluster_size=0). Found {len(points_for_algo_input)} points, needed {min_samples_for_algo}.")
                    if should_generate_plot_for_this_iteration and ax3: ax3.text(0,0,0,f'No Points/k=0 for {algorithm_name}',c='gray')

                if should_generate_plot_for_this_iteration and ax3:
                    if agent_traj_x:
                        ax3.plot(agent_traj_x,agent_traj_z,zs=0,zdir='z',marker='.',linestyle='-',color='blue',label='Trajectory')
                        ax3.scatter(agent_traj_x[-1],agent_traj_z[-1],zs=0,zdir='z',s=60,c='magenta',marker='*',label='Last Pose')

                    ax3_title_text = f"Detected ({detected_obstacle_count} BBoxes) by {algorithm_name}\n"
                    if algorithm_name == "DBSCAN": ax3_title_text += f"eps={model_params['eps']}, min_s={model_params['min_samples']}, Y-Scl={y_scale_factor_algo if y_scale_factor_algo != 1.0 else 'N/A'}"
                    elif algorithm_name == "HDBSCAN": ax3_title_text += f"min_cs={model_params['min_cluster_size']}, min_s={model_params.get('min_samples', 'def')}, eps_sel={model_params.get('cluster_selection_epsilon', 'def')}, Y-Scl={y_scale_factor_algo if y_scale_factor_algo != 1.0 else 'N/A'}"
                    elif algorithm_name == "KMEANS": ax3_title_text += f"K={dynamic_k_for_kmeans}"
                    elif algorithm_name == "GMM": ax3_title_text += f"K={dynamic_n_for_gmm}, Cov={model_params['covariance_type']}"
                    ax3.set_title(ax3_title_text, fontsize=9)
                    ax3.set_xlabel("X"); ax3.set_ylabel("Z (Depth)"); ax3.set_zlabel("Y (Height)")
                    ax3.set_xlim(-limit,limit); ax3.set_ylim(-limit,limit); ax3.set_zlim(0,10)
                    ax3.view_init(elev=30.,azim=-60)
                    if detected_obstacle_count > 0 or agent_traj_x: ax3.legend(fontsize='small')

                gt_boxes_info_for_eval_current_method = [gt_info.copy() for gt_info in gt_boxes_info_for_eval]
                for gt_box_eval in gt_boxes_info_for_eval_current_method: gt_box_eval['is_detected'] = False 

                tp = 0; fp = 0
                for p_box in predicted_boxes_info: p_box['is_tp'] = False

                if current_eval_method == "3D_IOU":
                    temp_gt_matched_by_pred_idx = [-1] * len(gt_boxes_info_for_eval_current_method)
                    pred_box_iou_with_gt = [0.0] * len(predicted_boxes_info)
                    for pred_idx, pred_box_info in enumerate(predicted_boxes_info):
                        best_iou_for_this_pred = 0; best_gt_eval_idx_for_this_pred = -1
                        for gt_eval_idx, gt_box_eval_info in enumerate(gt_boxes_info_for_eval_current_method):
                            iou = calculate_3d_iou(pred_box_info['min_coords'], pred_box_info['max_coords'], gt_box_eval_info['min_coords_3d'], gt_box_eval_info['max_coords_3d'])
                            if iou > best_iou_for_this_pred: best_iou_for_this_pred = iou; best_gt_eval_idx_for_this_pred = gt_eval_idx
                        if best_iou_for_this_pred >= IOU_THRESHOLD and best_gt_eval_idx_for_this_pred != -1:
                            gt_idx_to_match = best_gt_eval_idx_for_this_pred
                            if temp_gt_matched_by_pred_idx[gt_idx_to_match] == -1 or \
                               best_iou_for_this_pred > pred_box_iou_with_gt[temp_gt_matched_by_pred_idx[gt_idx_to_match]]:
                                if temp_gt_matched_by_pred_idx[gt_idx_to_match] != -1:
                                    predicted_boxes_info[temp_gt_matched_by_pred_idx[gt_idx_to_match]]['is_tp'] = False
                                predicted_boxes_info[pred_idx]['is_tp'] = True
                                pred_box_iou_with_gt[pred_idx] = best_iou_for_this_pred
                                gt_boxes_info_for_eval_current_method[gt_idx_to_match]['is_detected'] = True
                                temp_gt_matched_by_pred_idx[gt_idx_to_match] = pred_idx
                    tp = sum(1 for p_box in predicted_boxes_info if p_box['is_tp'])
                    fp = len(predicted_boxes_info) - tp
                elif current_eval_method == "2D_CENTER_IN_GT_AREA":
                    for pred_idx, pred_box_info in enumerate(predicted_boxes_info):
                        pred_center_x = (pred_box_info['min_coords'][0] + pred_box_info['max_coords'][0]) / 2
                        pred_center_z = (pred_box_info['min_coords'][2] + pred_box_info['max_coords'][2]) / 2
                        for gt_eval_idx, gt_box_eval_info in enumerate(gt_boxes_info_for_eval_current_method):
                            gt_min_x, gt_min_z = gt_box_eval_info['min_coords_2d_xz']
                            gt_max_x, gt_max_z = gt_box_eval_info['max_coords_2d_xz']
                            if (gt_min_x <= pred_center_x <= gt_max_x) and (gt_min_z <= pred_center_z <= gt_max_z):
                                if not gt_boxes_info_for_eval_current_method[gt_eval_idx]['is_detected']:
                                    predicted_boxes_info[pred_idx]['is_tp'] = True
                                    gt_boxes_info_for_eval_current_method[gt_eval_idx]['is_detected'] = True
                                    break
                    tp = sum(1 for p_box in predicted_boxes_info if p_box['is_tp'])
                    fp = len(predicted_boxes_info) - tp
                elif current_eval_method == "2D_AREA_OVERLAP_RATIO":
                    gt_matched_flags = [False] * len(gt_boxes_info_for_eval_current_method)
                    for pred_idx, pred_box_info in enumerate(predicted_boxes_info):
                        pred_min_xz = (pred_box_info['min_coords'][0], pred_box_info['min_coords'][2])
                        pred_max_xz = (pred_box_info['max_coords'][0], pred_box_info['max_coords'][2])
                        best_overlap_for_this_pred = 0; best_matching_gt_idx = -1
                        for gt_eval_idx, gt_box_eval_info in enumerate(gt_boxes_info_for_eval_current_method):
                            overlap_ratio = calculate_2d_xz_overlap_ratio(pred_min_xz, pred_max_xz, gt_box_eval_info['min_coords_2d_xz'], gt_box_eval_info['max_coords_2d_xz'])
                            if overlap_ratio > best_overlap_for_this_pred:
                                best_overlap_for_this_pred = overlap_ratio; best_matching_gt_idx = gt_eval_idx

                        if best_overlap_for_this_pred >= AREA_OVERLAP_RATIO_THRESHOLD and best_matching_gt_idx != -1:
                            if not gt_matched_flags[best_matching_gt_idx]:
                                predicted_boxes_info[pred_idx]['is_tp'] = True
                                gt_boxes_info_for_eval_current_method[best_matching_gt_idx]['is_detected'] = True
                                gt_matched_flags[best_matching_gt_idx] = True
                    tp = sum(1 for p_box in predicted_boxes_info if p_box['is_tp'])
                    fp = len(predicted_boxes_info) - tp

                fn = sum(1 for gt_box_eval_info in gt_boxes_info_for_eval_current_method if not gt_box_eval_info['is_detected'])

                if should_generate_plot_for_this_iteration and ax2:
                    gnd_size=gnd_info['size']; ax2.add_patch(patches.Rectangle((-gnd_size/2,-gnd_size/2),gnd_size,gnd_size,ec='gray',fc='none',ls='--'))
                    tp_label_added=False; fn_label_added=False; other_gt_label_added = False; fp_label_added = False
                    for idx, row in current_gt_obstacle_df.iterrows():
                        current_label = None; rect_color = 'lightgrey'; rect_alpha = 0.3
                        is_relevant_gt = False; gt_is_detected_for_plot = False
                        for gt_eval_info in gt_boxes_info_for_eval_current_method:
                            if gt_eval_info['original_gt_idx'] == idx:
                                is_relevant_gt = True; gt_is_detected_for_plot = gt_eval_info['is_detected']; break
                        if is_relevant_gt:
                            if gt_is_detected_for_plot: rect_color = 'forestgreen'; rect_alpha = 0.7
                            else: rect_color = 'crimson'; rect_alpha = 0.6
                            if gt_is_detected_for_plot and not tp_label_added: current_label = 'TP GT (Detected)'; tp_label_added = True
                            elif not gt_is_detected_for_plot and not fn_label_added: current_label = 'FN GT (Missed)'; fn_label_added = True
                        elif not other_gt_label_added: current_label = 'Other GT (Out of Range)'; other_gt_label_added = True
                        ax2.add_patch(patches.Rectangle((row['center_x']-row['width']/2,row['center_z']-row['depth']/2), row['width'],row['depth'],ec='k',fc=rect_color,alpha=rect_alpha,label=current_label))

                    for pred_box_info in predicted_boxes_info:
                        if not pred_box_info['is_tp']:
                            pred_min_x = pred_box_info['min_coords'][0]; pred_max_x = pred_box_info['max_coords'][0]
                            pred_min_z = pred_box_info['min_coords'][2]; pred_max_z = pred_box_info['max_coords'][2]
                            fp_width = pred_max_x - pred_min_x; fp_depth = pred_max_z - pred_min_z
                            current_label = None
                            if not fp_label_added: current_label = 'FP Prediction'; fp_label_added = True
                            ax2.add_patch(patches.Rectangle((pred_min_x, pred_min_z), fp_width, fp_depth, edgecolor='darkorange', facecolor='none', hatch='//', alpha=0.7, linestyle='--', label=current_label))

                    ax2.scatter(0,0,s=100,c='red',marker='x',label='Start'); ax2.set_title(f"GT Map (Eval: {current_eval_method})")
                    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Z (m)"); ax2.grid(True,ls='--',alpha=0.6); ax2.set_aspect('equal','box')
                    ax2.legend(fontsize='small'); ax2.set_xlim(-limit,limit); ax2.set_ylim(-limit,limit)

                if should_generate_plot_for_this_iteration and fig_algo:
                    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
                    precision=tp/(tp+fp) if (tp+fp)>0 else 0; recall=tp/(tp+fn) if (tp+fn)>0 else 0
                    f1_score=2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0

                    eval_method_str_display = current_eval_method.replace("_", " ")
                    threshold_str = ""
                    if current_eval_method == "3D_IOU": threshold_str = f" (IoU Thresh: {IOU_THRESHOLD})"
                    elif current_eval_method == "2D_AREA_OVERLAP_RATIO": threshold_str = f" (Overlap Thresh: {AREA_OVERLAP_RATIO_THRESHOLD})"

                    eval_text = f"Algorithm: {algorithm_name}, Eval: {eval_method_str_display}{threshold_str}, LiDAR Range:{LIDAR_RANGE}m\n"
                    eval_text += f"  Relevant GT: {len(gt_boxes_info_for_eval_current_method)}, Predicted: {len(predicted_boxes_info)}\n"
                    eval_text += f"  TP: {tp}, FP: {fp}, FN: {fn}\n  Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1_score:.3f}\n"
                    eval_text += f"  Clustering Time: {clustering_time_sec:.4f} sec"
                    fig_algo.text(0.5, 0.01, eval_text, ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7)) # y 위치 및 폰트 크기 조정
                else: 
                    precision=tp/(tp+fp) if (tp+fp)>0 else 0; recall=tp/(tp+fn) if (tp+fn)>0 else 0
                    f1_score=2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0


                all_algorithms_results_summary_for_all_eval_methods.append({
                    "Algorithm": algorithm_name,
                    "Params": str(model_params) + (f", YScl:{y_scale_factor_algo}" if y_scale_factor_algo!=1.0 and algorithm_name in ["DBSCAN", "HDBSCAN"] else ""),
                    "EvaluationMethod": current_eval_method,
                    "TP": tp, "FP": fp, "FN": fn,
                    "Precision": round(precision,3), "Recall": round(recall,3), "F1-Score": round(f1_score,3),
                    "ClusteringTime_sec": round(clustering_time_sec, 4), # 시간 추가
                    "GT_Relevant": len(gt_boxes_info_for_eval_current_method), "Predicted_Boxes": detected_obstacle_count
                })

        print("\n\n--- Final Performance Summary (All Evaluation Methods) ---")
        summary_df_all_methods = pd.DataFrame(all_algorithms_results_summary_for_all_eval_methods)
        print(summary_df_all_methods.to_string())

        if show_plots:
            if len(plt.get_fignums()) > 0:
                print(f"DEBUG: Displaying {len(plt.get_fignums())} Matplotlib figure(s) for '2D_AREA_OVERLAP_RATIO' method.")
                plt.show(block=True)
                print("DEBUG: plt.show(block=True) has returned.")
            else:
                print("DEBUG: No figures were generated for display (as per evaluation method filtering or no 'show_plots').")
        return summary_df_all_methods

    except Exception as e_plot:
        print(f"Error during Matplotlib plotting or evaluation: {e_plot}")
        import traceback
        traceback.print_exc()
        return None

def save_performance_summary_to_csv(summary_df, filename="simulation_performance_summary.csv"):
    if summary_df is None or summary_df.empty:
        print("No summary data to save.")
        return

    summary_df['Timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    summary_df['Seed'] = current_obstacle_seed
    summary_df['TotalSimTime'] = TOTAL_SIMULATION_DURATION
    summary_df['LiDAR_Noise_Enabled'] = LIDAR_NOISE_ENABLED # 노이즈 설정 추가
    summary_df['LiDAR_Noise_Level'] = LIDAR_NOISE_LEVEL if LIDAR_NOISE_ENABLED else 0.0 # 노이즈 설정 추가

    summary_df['NumInitialObstacles'] = NUM_OBSTACLES
    summary_df['AgentSpeed'] = AGENT_SPEED
    summary_df['LiDAR_Range'] = LIDAR_RANGE
    summary_df['PaddingCells'] = OBSTACLE_PADDING_CELLS


    try:
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
            combined_df.to_csv(filename, index=False)
            print(f"Performance summary appended to {filename}")
        else:
            summary_df.to_csv(filename, index=False, header=True)
            print(f"Performance summary saved to new file: {filename}")
    except Exception as e:
        print(f"Error saving performance summary to CSV: {e}")

info_text = Text(origin=(-0.5,0.5), scale=(0.8,0.8), x=-0.5*window.aspect_ratio+0.02, y=0.48, text="Initializing...")

def input(key):
    global scan_history,obstacle_df,movement_states,rotation_states, current_obstacle_seed
    global simulation_running, simulation_time_elapsed, next_dynamic_event_idx, target_destination_world, replanning_needed, current_path_waypoints, current_waypoint_idx, current_destination_timer
    global current_grid_map, path_visualization_entities
    global LIDAR_NOISE_ENABLED # 전역 변수 접근

    if key == 'k':
        if not simulation_running:
            simulation_running = True
            simulation_time_elapsed = 0.0
            next_dynamic_event_idx = 0
            target_destination_world = None
            replanning_needed = True
            current_path_waypoints = []
            current_waypoint_idx = 0
            current_destination_timer = 0.0
            scan_history = []

            clear_obstacles()
            generate_obstacles(current_obstacle_seed)
            current_grid_map = GridMap(AREA_SIZE, GRID_CELL_SIZE, obstacle_df, padding_cells=OBSTACLE_PADDING_CELLS)
            visualize_path([])
            print("Autonomous driving simulation started!")
        else:
            print("Autonomous driving simulation stopped by user.")
            simulation_running = False
            visualize_path([])
            if scan_history:
                print("Saving results after manual stop...")
                summary_df_results = evaluate_and_plot_algorithms(scan_history, obstacle_df, ground_info, show_plots=True) 
                if summary_df_results is not None and not summary_df_results.empty:
                    save_performance_summary_to_csv(summary_df_results)
                else:
                    print("No performance summary to save after manual stop.")
            else:
                print("No scan history to evaluate after manual stop.")
            target_destination_world = None
            current_path_waypoints = []
        return

    if simulation_running:
        if key == 'escape': application.quit()
        return

    if key == 'v': # 노이즈 토글 키
        LIDAR_NOISE_ENABLED = not LIDAR_NOISE_ENABLED
        print(f"LiDAR Noise {'ENABLED' if LIDAR_NOISE_ENABLED else 'DISABLED'}")
        return


    if key in ('0', '4', '5', '6', '7', '8', '9'):
        new_seed_offset = int(key)
        new_seed = RANDOM_OBSTACLE_SEED
        if key == '0': new_seed = RANDOM_OBSTACLE_SEED
        else: new_seed = RANDOM_OBSTACLE_SEED + new_seed_offset

        if new_seed == current_obstacle_seed and key != '0':
             print(f"Already using the same environment seed ({current_obstacle_seed}).")
        else:
            clear_obstacles()
            generate_obstacles(new_seed); scan_history = []
            print(f"Obstacles reset (New Seed: {current_obstacle_seed}). Scan history cleared.")
            current_grid_map = GridMap(AREA_SIZE, GRID_CELL_SIZE, obstacle_df, padding_cells=OBSTACLE_PADDING_CELLS)
            visualize_path([])
    elif key == 'n': add_dynamic_obstacle(count=1)
    elif key == 'x': remove_dynamic_obstacle(count=1, strategy="random")
    elif key in ['w','w hold']: movement_states['forward']=True
    elif key=='w up': movement_states['forward']=False
    elif key in ['s','s hold']: movement_states['backward']=True
    elif key=='s up': movement_states['backward']=False
    elif key in ['a','a hold']: movement_states['left']=True
    elif key=='a up': movement_states['left']=False
    elif key in ['d','d hold']: movement_states['right']=True
    elif key=='d up': movement_states['right']=False
    elif key in ['q','q hold']: rotation_states['left']=True
    elif key=='q up': rotation_states['left']=False
    elif key in ['e','e hold']: rotation_states['right']=True
    elif key=='e up': rotation_states['right']=False
    elif key in ['m','M']:
        if scan_history:
            print("DEBUG: 'M' key pressed. Calling evaluate_and_plot_algorithms with show_plots=True (only 2D_AREA_OVERLAP_RATIO will plot)")
            summary_df_results = evaluate_and_plot_algorithms(scan_history,obstacle_df,ground_info, show_plots=True)
        else:
            print("No scan history to evaluate. Run the simulation first (press 'K').")
    elif key == 'l':
        if not simulation_running and scan_history:
            print("Manually evaluating and saving results (no plots shown)...")
            summary_df_results = evaluate_and_plot_algorithms(scan_history, obstacle_df, ground_info, show_plots=False)
            if summary_df_results is not None and not summary_df_results.empty:
                save_performance_summary_to_csv(summary_df_results)
            else:
                print("No performance summary to save from manual evaluation.")
        elif simulation_running:
            print("Cannot manually evaluate while simulation is running. Stop with 'K' first.")
        else:
            print("No scan history to evaluate manually.")


    elif key in ['c','C']: scan_history=[]; print("Scan history cleared.")
    elif key in ['p','P']:
        ts=time.strftime('%Y%m%d_%H%M%S'); lr_str=str(LIDAR_RANGE).replace('.','_'); ys_str=str(Y_COORD_SCALE_FACTOR).replace('.','_')
        eps_str=str(DBSCAN_EPS).replace('.','_'); min_s_str=str(DBSCAN_MIN_SAMPLES)
        base_filename = f"s{current_obstacle_seed}_lr{lr_str}_{ts}"
        if LIDAR_NOISE_ENABLED: base_filename += f"_noise{str(LIDAR_NOISE_LEVEL).replace('.', '_')}"
        if scan_history: fn_hist=f"scan_{base_filename}.pkl"; pickle.dump(scan_history,open(fn_hist,'wb')); print(f"Scan history saved: {fn_hist}")
        if obstacle_df is not None and not obstacle_df.empty : fn_obs=f"gt_{base_filename}.csv"; obstacle_df.to_csv(fn_obs,index=False); print(f"GT saved: {fn_obs}")
    elif key=='escape': application.quit()

def update():
    global scan_timer, scan_history, movement_states, rotation_states, current_obstacle_seed
    global simulation_running, simulation_time_elapsed, next_dynamic_event_idx, target_destination_world, replanning_needed
    global current_path_waypoints, current_waypoint_idx, current_grid_map, current_destination_timer, path_visualization_entities

    # 1. 에이전트 이동 및 회전 (자율 또는 수동)
    if simulation_running:
        simulation_time_elapsed += time.dt
        current_destination_timer += time.dt

        if simulation_time_elapsed >= TOTAL_SIMULATION_DURATION and simulation_running:
            print("\nTotal simulation time automatically ended.")
            simulation_running = False
            visualize_path([])
            if scan_history:
                print("Automatically saving results (plots for 2D_AREA_OVERLAP_RATIO if show_plots=True)...")
                summary_df_results = evaluate_and_plot_algorithms(scan_history, obstacle_df, ground_info, show_plots=False)
                if summary_df_results is not None and not summary_df_results.empty:
                    save_performance_summary_to_csv(summary_df_results)
                else:
                    print("No performance summary to save.")
            else:
                print("No scan history to evaluate.")

            target_destination_world = None
            current_path_waypoints = []

        if simulation_running:
            if next_dynamic_event_idx < len(DYNAMIC_EVENT_SCHEDULE):
                event_time, event_type, event_params = DYNAMIC_EVENT_SCHEDULE[next_dynamic_event_idx]
                if simulation_time_elapsed >= event_time:
                    print(f"\n[Dynamic Event Triggered] Time: {simulation_time_elapsed:.1f}s, Type: {event_type}")
                    if event_type == "ADD_OBSTACLE":
                        add_dynamic_obstacle(**event_params)
                    elif event_type == "REMOVE_OBSTACLE":
                        remove_dynamic_obstacle(**event_params)
                    elif event_type == "CHANGE_MAP_SEED":
                        new_map_seed = current_obstacle_seed + event_params.get("new_seed_offset", 100)
                        print(f"  Changing map seed: {current_obstacle_seed} -> {new_map_seed}")
                        clear_obstacles()
                        generate_obstacles(new_map_seed)
                        replanning_needed = True

                    next_dynamic_event_idx += 1

            if target_destination_world is None or not current_path_waypoints or replanning_needed:
                if (target_destination_world is None or \
                   (current_path_waypoints and current_waypoint_idx >= len(current_path_waypoints)) or \
                   current_destination_timer > TARGET_DESTINATION_REACH_TIMEOUT) :

                    generated_valid_target = False
                    for _ in range(MAX_TARGET_GENERATION_ATTEMPTS):
                        candidate_x = random.uniform(-AREA_SIZE/2.2, AREA_SIZE/2.2)
                        candidate_z = random.uniform(-AREA_SIZE/2.2, AREA_SIZE/2.2)
                        candidate_target = Vec3(candidate_x, AGENT_HEIGHT, candidate_z)
                        dist_to_candidate = distance(agent.position, candidate_target)
                        if MIN_TARGET_DISTANCE <= dist_to_candidate <= MAX_TARGET_DISTANCE:
                            target_destination_world = candidate_target
                            generated_valid_target = True
                            break

                    if generated_valid_target:
                        print(f"  New target destination (dist: {distance(agent.position, target_destination_world):.1f}): ({target_destination_world.x:.1f}, {target_destination_world.z:.1f})")
                        replanning_needed = True
                        current_destination_timer = 0.0
                    else:
                        print(f"  Could not find a suitable new target within distance constraints after {MAX_TARGET_GENERATION_ATTEMPTS} attempts. Will try again.")
                        target_destination_world = None

                if replanning_needed and target_destination_world and current_grid_map:
                    print(f"  Replanning path... Current: ({agent.x:.1f},{agent.z:.1f}) -> Target: ({target_destination_world.x:.1f},{target_destination_world.z:.1f})")
                    current_path_waypoints = a_star_search(current_grid_map, (agent.x, agent.z), (target_destination_world.x, target_destination_world.z))
                    current_waypoint_idx = 0
                    replanning_needed = False
                    visualize_path(current_path_waypoints)
                    if not current_path_waypoints:
                        print("  Path planning failed. Waiting for next attempt or new target.")
                        target_destination_world = None
                        visualize_path([])
                    else:
                        print(f"  Path generated: {len(current_path_waypoints)} waypoints.")


            if current_path_waypoints and current_waypoint_idx < len(current_path_waypoints):
                target_wp = current_path_waypoints[current_waypoint_idx]
                direction_to_wp = (target_wp - agent.position).normalized()

                target_rotation_y = math.degrees(math.atan2(direction_to_wp.x, direction_to_wp.z))
                angle_diff = normalize_angle(math.radians(target_rotation_y - agent.rotation_y))
                agent.rotation_y += math.degrees(angle_diff) * ROTATION_SPEED * time.dt * 0.1

                if abs(angle_diff) < math.radians(15):
                    move_amount = AGENT_SPEED * time.dt
                    original_agent_pos = agent.position
                    agent.position += Vec3(direction_to_wp.x, 0, direction_to_wp.z).normalized() * move_amount

                    hit_info_move = agent.intersects(traverse_target=scene)
                    if hit_info_move.hit and hasattr(hit_info_move.entity,'tag') and hit_info_move.entity.tag==OBSTACLE_TAG:
                        agent.position = original_agent_pos
                        replanning_needed = True
                        current_path_waypoints = []
                        visualize_path([])

                if distance(agent.position, target_wp) < WAYPOINT_THRESHOLD:
                    current_waypoint_idx += 1
                    if current_waypoint_idx >= len(current_path_waypoints):
                        print("  All waypoints reached for the current target.")
                        visualize_path([])

    else: # 수동 조작 모드
        original_pos=agent.position; total_dx,total_dz=0.0,0.0; speed_dt=MANUAL_AGENT_SPEED*time.dt
        if movement_states['forward']: total_dx+=agent.forward.x*speed_dt; total_dz+=agent.forward.z*speed_dt
        if movement_states['backward']: total_dx-=agent.forward.x*speed_dt; total_dz-=agent.forward.z*speed_dt
        if movement_states['left']: total_dx-=agent.right.x*speed_dt; total_dz-=agent.right.z*speed_dt
        if movement_states['right']: total_dx+=agent.right.x*speed_dt; total_dz+=agent.right.z*speed_dt
        current_x,current_z=agent.x,agent.z
        agent.x+=total_dx; hit_info_move=agent.intersects(traverse_target=scene)
        if hit_info_move.hit and hasattr(hit_info_move.entity,'tag') and hit_info_move.entity.tag==OBSTACLE_TAG: agent.x=current_x
        agent.z+=total_dz; hit_info_move=agent.intersects(traverse_target=scene)
        if hit_info_move.hit and hasattr(hit_info_move.entity,'tag') and hit_info_move.entity.tag==OBSTACLE_TAG: agent.z=current_z
        agent.y=AGENT_HEIGHT
        if rotation_states['left']: agent.rotation_y-=ROTATION_SPEED*time.dt
        if rotation_states['right']: agent.rotation_y+=ROTATION_SPEED*time.dt

    # 2. LiDAR 스캔 (에이전트 이동/회전 *후에* 실행)
    scan_timer += time.dt
    perform_scan_this_frame = False
    current_actual_scan_interval = SCAN_INTERVAL if simulation_running else MANUAL_SCAN_INTERVAL
    if scan_timer >= current_actual_scan_interval:
        perform_scan_this_frame = True
        scan_timer -= current_actual_scan_interval

    if perform_scan_this_frame:
        frozen_agent_world_pos = agent.world_position
        frozen_agent_world_rotation_y_deg = agent.world_rotation_y
        current_pose_for_history = (frozen_agent_world_pos.x, frozen_agent_world_pos.z, math.radians(frozen_agent_world_rotation_y_deg))
        add_this_scan = True
        if scan_history:
            last_pose = scan_history[-1]['pose']
            pos_change = math.sqrt((current_pose_for_history[0] - last_pose[0])**2 + (current_pose_for_history[1] - last_pose[1])**2)
            rot_diff_rad = abs(normalize_angle(current_pose_for_history[2] - last_pose[2]))
            if pos_change < MIN_MOVEMENT_THRESHOLD_FOR_NEW_SCAN and rot_diff_rad < math.radians(MIN_ROTATION_THRESHOLD_FOR_NEW_SCAN_DEG):
                add_this_scan = False
        if add_this_scan:
            rel_pts = generate_relative_lidar_map_3d_at_pose(frozen_agent_world_pos, frozen_agent_world_rotation_y_deg)
            if rel_pts.shape[0] > 0:
                scan_history.append({'pose': current_pose_for_history, 'relative_points': rel_pts.tolist()})

    # 3. 시각화 및 정보 업데이트 (항상 실행)
    if VISUALIZE_LIDAR_RAYS: update_lidar_visualization()
    else:
        global lidar_lines
        for line in lidar_lines: destroy(line)
        lidar_lines.clear()

    sim_status_text = "AutoDriving" if simulation_running else "ManualMode"
    if simulation_running:
        sim_status_text += f" (Time: {simulation_time_elapsed:.1f}/{TOTAL_SIMULATION_DURATION:.0f}Sec)"
        if target_destination_world:
             sim_status_text += f"\n Target: ({target_destination_world.x:.1f},{target_destination_world.z:.1f})"
             if current_path_waypoints and current_waypoint_idx < len(current_path_waypoints):
                 sim_status_text += f" WP: {current_waypoint_idx+1}/{len(current_path_waypoints)}"
        sim_status_text += f" Event: {next_dynamic_event_idx}/{len(DYNAMIC_EVENT_SCHEDULE)}"

    noise_status_text = f"LiDAR Noise: {'ON' if LIDAR_NOISE_ENABLED else 'OFF'} (Lvl:{LIDAR_NOISE_LEVEL if LIDAR_NOISE_ENABLED else 'N/A'})"
    info_text.text=( f"Location:({agent.x:.1f},{agent.z:.1f}) RotY:{agent.rotation_y:.0f}° ScanNum:{len(scan_history)}\n"
                    f"Simulation: {sim_status_text} (Seed :{current_obstacle_seed})\n"
                    f"{noise_status_text} | Eval Methods: {', '.join(POSSIBLE_EVALUATION_METHODS)}\n"
                    f"K:AutoDrive V:NoiseOnOff M:Eval&Plot L:Eval&Save C:ScanReset P:SaveRaw\n"
                    f"0,4-9:ObsSetting N:AddObs X:DelObs (ManualMode)")
    info_text.x=-0.5*window.aspect_ratio+0.02; info_text.y=0.36 # 텍스트 위치 약간 조정
    if mouse.locked:mouse.locked=False
    if not mouse.visible:mouse.visible=True

# --- 시뮬레이션 시작 ---
if __name__ == '__main__':
    create_initial_environment()
    generate_obstacles(RANDOM_OBSTACLE_SEED)
    current_grid_map = GridMap(AREA_SIZE, GRID_CELL_SIZE, obstacle_df, padding_cells=OBSTACLE_PADDING_CELLS)
    app.run()