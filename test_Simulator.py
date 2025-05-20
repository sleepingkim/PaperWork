# -*- coding: utf-8 -*-
# 필요한 라이브러리들을 임포트합니다.
import sys # 시스템 관련 모듈
import os # 운영체제 관련 모듈
import time # 시간 관련 모듈

# (주석 처리됨) Ursina 라이브러리 경로를 수동으로 설정해야 할 경우 사용합니다.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# ursina_path = os.path.join(current_dir, '..')
# sys.path.append(ursina_path)

try:
    # Ursina: 3D 게임 엔진 및 시뮬레이션 환경 제공
    from ursina import *
    # Scikit-learn: DBSCAN 클러스터링 알고리즘 사용
    from sklearn.cluster import DBSCAN
    # NumPy: 다차원 배열 및 수학 연산 지원 (필수)
    import numpy as np
    # Matplotlib: 데이터 시각화 (특히 3D 플롯)
    from mpl_toolkits.mplot3d import Axes3D # 3D 플롯 위해 추가
    import open3d as o3d # Open3D: 3D 데이터 처리 (RANSAC 등)
except ImportError as e:
    # 필수 라이브러리가 설치되지 않았을 경우 오류 메시지 출력 및 종료
    t_now = time.strftime("%Y-%m-%d %H:%M:%S") # 현재 시간 포함
    print(f"[{t_now}] 오류: 필수 라이브러리 임포트 실패: {e}")
    print(f"[{t_now}] Ursina, Scikit-learn, NumPy, Pandas, Matplotlib, Open3D 라이브러리가 설치되어 있는지 확인하세요.")
    print(f"[{t_now}] 설치 명령어 예시: pip install ursina scikit-learn numpy pandas matplotlib open3d")
    sys.exit(1) # 프로그램 종료
except Exception as e:
    # 기타 예기치 않은 임포트 오류 발생 시 처리
    t_now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_now}] 임포트 중 예기치 않은 오류 발생: {e}")
    sys.exit(1)

# --- 추가 필수 라이브러리 ---
import random # 무작위 값 생성 (장애물 배치 등)
import math   # 수학 함수 사용 (삼각 함수, 각도 변환 등)
import matplotlib.pyplot as plt # 2D 및 3D 그래프 생성
import matplotlib.patches as patches # 2D 도형(사각형 등) 그리기 (Ground Truth 맵)
import pandas as pd # 데이터프레임 사용 (장애물 정보 저장)
import pickle # 파이썬 객체 직렬화/역직렬화 (scan_history 저장/로드)

# --- 시뮬레이션 환경 설정값 ---
RANDOM_OBSTACLE_SEED = 1  # 장애물 생성을 위한 랜덤 시드 (이 값을 변경하면 다른 배치 생성)

NUM_OBSTACLES = 25      # 생성할 장애물 개수 (시도할 최대 개수)
AREA_SIZE = 50          # 시뮬레이션 영역의 크기 (정사각형 영역의 한 변의 길이)
AGENT_SPEED = 10         # 에이전트(로봇)의 이동 속도 (초당 단위)
ROTATION_SPEED = 100    # 에이전트의 회전 속도 (초당 각도)
AGENT_HEIGHT = 0.5      # 에이전트의 높이 (Y 좌표, 바닥으로부터의 높이)
LIDAR_RANGE = 15        # 라이다 센서의 최대 감지 거리 (미터 단위)
LIDAR_FOV = 360        # 라이다의 수평 시야각 (도)
NUM_LIDAR_RAYS = 144     # 라이다의 수평 해상도 (수평 방향으로 발사되는 광선 수)
LIDAR_VERTICAL_FOV = 45 # 라이다의 수직 시야각 (도)
NUM_LIDAR_CHANNELS = 8  # 라이다의 수직 채널 수 (수직 방향 스캔 라인 수)
LIDAR_VIS_HEIGHT = 0.6  # 라이다 센서의 시각화 및 스캔 시작 높이 (에이전트 바닥 기준 오프셋)
LIDAR_COLOR = color.cyan # 라이다 광선 기본 색상 (충돌하지 않았을 때)
OBSTACLE_TAG = "obstacle" # 장애물 엔티티를 식별하기 위한 태그 문자열
SCAN_INTERVAL = 1.0     # 라이다 스캔 실행 간격 (초)
VISUALIZE_LIDAR_RAYS = False # LiDAR 광선 실시간 시각화 여부 (True로 바꾸면 켜짐)

# --- 클러스터링 알고리즘 (DBSCAN) 설정값 ---
DBSCAN_EPS = 1        # DBSCAN: 클러스터 내 점들 간의 최대 거리 (epsilon)
DBSCAN_MIN_SAMPLES = 3 # DBSCAN: 클러스터를 형성하기 위한 최소 점 개수 (MinPts)
Y_COORD_SCALE_FACTOR = 0.2 # DBSCAN 입력 전 Y축 스케일링 팩터 (값이 작을수록 높이 차이에 덜 민감)

# --- RANSAC 지면 분할 설정값 ---
RANSAC_DISTANCE_THRESHOLD = 0.5 # RANSAC: 지면 평면으로부터 포인트가 지면으로 간주될 최대 거리
RANSAC_N_POINTS = 3             # RANSAC: 평면 추정을 위해 무작위로 선택할 최소 포인트 수
RANSAC_NUM_ITERATIONS = 100     # RANSAC: 알고리즘 반복 횟수
VISUALIZE_RANSAC_SEGMENTATION = False # True로 설정 시 Open3D 창으로 RANSAC 분할 결과 표시

# --- 성능 평가 설정값 ---
IOU_THRESHOLD = 0.5 # TP(True Positive) 판정을 위한 3D IoU 임계값 (3D_IOU 방식 사용 시)
EVALUATION_METHOD = "2D_AREA_OVERLAP_RATIO" # 정확도 평가 방식: "3D_IOU", "2D_CENTER_IN_GT_AREA", "2D_AREA_OVERLAP_RATIO"
AREA_OVERLAP_RATIO_THRESHOLD = 0.3 # XZ 평면상 GT 영역 대비 겹침 비율 임계값 (2D_AREA_OVERLAP_RATIO 방식 사용 시)


# --- Ursina 애플리케이션 설정 ---
app = Ursina(
    title="3D LiDAR Sim - 상세 주석 및 기능 종합", # 창 제목 설정
    borderless=False, # 창 테두리 표시 여부
)

# --- 마우스 설정 ---
mouse.visible = True # 마우스 커서 보이기
mouse.locked = False # 마우스 커서 잠금 해제 (카메라 제어 비활성화)

# --- 전역 변수 초기화 ---
scan_timer = 0.0 # 다음 라이다 스캔까지 남은 시간을 추적하는 타이머
scan_history = [] # 각 스캔 시점의 에이전트 자세와 감지된 상대 좌표점들을 저장하는 리스트
obstacle_df = None # 실제 장애물 정보를 저장할 Pandas DataFrame (Ground Truth용)
ground_info = {'size': AREA_SIZE, 'center_x': 0, 'center_z': 0} # 시뮬레이션 영역(바닥) 정보
movement_states = {'forward': False, 'backward': False, 'left': False, 'right': False} # 이동 상태
rotation_states = {'left': False, 'right': False} # 회전 상태


# --- 환경 요소 생성 (바닥, 하늘, 장애물) ---
# 바닥 생성: 평면 모델, 지정된 크기, 밝은 회색, 텍스처 적용, 충돌체 설정
ground = Entity(model='plane', scale=(AREA_SIZE,1,AREA_SIZE), color=color.light_gray, texture='white_cube', texture_scale=(AREA_SIZE/2,AREA_SIZE/2), collider='box')
# 하늘 생성: 기본적인 하늘 배경
sky = Sky()

obstacles_list_for_df = []  # GT DataFrame용 장애물 정보 리스트
obstacles_entities = []     # Ursina 환경 내 장애물 엔티티들을 저장할 리스트
placed_obstacles_aabbs = [] # 성공적으로 배치된 장애물들의 AABB(축 정렬 경계 상자) 정보 저장용

min_obstacle_distance_from_spawn = 3.0 # 에이전트 시작 위치로부터 장애물까지의 최소 거리 (시작 시 충돌 방지)
max_placement_attempts_per_obstacle = 100 # 장애물 하나당 배치 최대 시도 횟수 (겹침 방지용)

# 랜덤 시드 설정 (장애물 생성 전 호출하여 항상 동일한 패턴으로 장애물 생성)
random.seed(RANDOM_OBSTACLE_SEED)
print(f"장애물 생성에 사용된 랜덤 시드: {RANDOM_OBSTACLE_SEED}")
print(f"{NUM_OBSTACLES}개의 장애물 생성 시도 (겹침 방지 및 시드 적용)...")

# 지정된 개수(NUM_OBSTACLES)만큼 장애물 생성 시도
for i in range(NUM_OBSTACLES):
    placed_successfully = False # 현재 장애물이 성공적으로 배치되었는지 여부
    # 겹치지 않는 위치를 찾기 위해 최대 시도 횟수만큼 반복
    for attempt in range(max_placement_attempts_per_obstacle):
        # 1. 후보 장애물 위치 및 크기 무작위 생성
        pos_x_cand = random.uniform(-AREA_SIZE / 2.8, AREA_SIZE / 2.8) # 영역 가장자리에서 약간 안쪽으로
        pos_z_cand = random.uniform(-AREA_SIZE / 2.8, AREA_SIZE / 2.8)
        
        scale_x_cand = random.uniform(0.5, 3.0) # 너비 (X축 크기)
        scale_y_cand = random.uniform(1.0, 5.0) # 높이 (Y축 크기)
        scale_z_cand = random.uniform(0.5, 3.0) # 깊이 (Z축 크기)
        pos_y_cand = scale_y_cand / 2 # 모델 중심이 바닥에 놓이도록 Y 좌표 설정

        # 2. 후보 장애물의 AABB(World Space) 계산
        cand_min_x = pos_x_cand - scale_x_cand/2
        cand_max_x = pos_x_cand + scale_x_cand/2
        cand_min_y = pos_y_cand - scale_y_cand/2 # 실제로는 0 (바닥에 붙어있음)
        cand_max_y = pos_y_cand + scale_y_cand/2 # 실제로는 scale_y_cand
        cand_min_z = pos_z_cand - scale_z_cand/2
        cand_max_z = pos_z_cand + scale_z_cand/2

        # 3. 기존에 배치된 장애물들과의 충돌(겹침) 검사
        overlap_found = False
        for placed_aabb in placed_obstacles_aabbs:
            # 두 AABB가 X, Y, Z 모든 축에서 겹치는지 확인
            x_overlap = (cand_max_x > placed_aabb['min_x'] and cand_min_x < placed_aabb['max_x'])
            y_overlap = (cand_max_y > placed_aabb['min_y'] and cand_min_y < placed_aabb['max_y']) # Y축(높이)도 고려
            z_overlap = (cand_max_z > placed_aabb['min_z'] and cand_min_z < placed_aabb['max_z'])
            if x_overlap and y_overlap and z_overlap:
                overlap_found = True # 겹침 발견
                break # 현재 후보는 겹치므로 더 이상 검사할 필요 없음
        
        # 4. 겹치지 않으면 장애물 생성 및 정보 저장
        if not overlap_found:
            obstacle_color = color.random_color() # 랜덤 시드에 의해 결정되는 색상
            obs_entity = Entity(
                model='cube', # 모델 형태: 큐브
                position=(pos_x_cand, pos_y_cand, pos_z_cand), # 위치 설정 (중심 기준)
                scale=(scale_x_cand, scale_y_cand, scale_z_cand), # 크기 설정
                color=obstacle_color, # 색상 설정
                collider='box', # 충돌체 형태: 박스 (물리적 상호작용용)
                tag=OBSTACLE_TAG, # 태그 설정 (라이다 감지용)
                name=f"obstacle_actual_{len(obstacles_entities)}" # 실제 생성된 순서대로 이름 부여
            )
            obstacles_entities.append(obs_entity) # 생성된 엔티티를 리스트에 추가
            
            # 배치된 장애물의 AABB 정보 저장 (겹침 검사용)
            placed_obstacles_aabbs.append({
                'min_x':cand_min_x, 'max_x':cand_max_x, 'min_y':cand_min_y,
                'max_y':cand_max_y, 'min_z':cand_min_z, 'max_z':cand_max_z,
            })

            # Ground Truth DataFrame용 정보 저장
            obstacles_list_for_df.append({
                'name': obs_entity.name, # 일관된 이름 사용
                'center_x': pos_x_cand, 'center_y': pos_y_cand, 'center_z': pos_z_cand, # 중심 좌표
                'width': scale_x_cand, 'height': scale_y_cand, 'depth': scale_z_cand,   # 크기
                'rotation_y': 0 # 현재는 회전 없음
            })
            placed_successfully = True # 현재 장애물 배치 성공
            break # 현재 장애물(i)에 대한 배치 시도 중단, 다음 장애물(i+1)로 넘어감
    
    if not placed_successfully:
        print(f"경고: 장애물 인덱스 {i}는 {max_placement_attempts_per_obstacle}번 시도 후에도 겹치지 않는 위치를 찾지 못했습니다.")

# Ground Truth 장애물 정보로 Pandas DataFrame 생성
obstacle_df = pd.DataFrame(obstacles_list_for_df)
print(f"실제 장애물 정보 DataFrame 생성 완료 ({len(obstacles_list_for_df)}개 생성됨).")


# --- 에이전트(로봇) 및 카메라 설정 ---
# 에이전트 생성: 구체 모델, 파란색, 초기 위치(0, AGENT_HEIGHT, 0), 충돌체 설정
agent = Entity(model='sphere', color=color.blue, position=(0,AGENT_HEIGHT,0), collider='sphere', scale=1)
# 카메라 설정:
camera.parent = agent        # 카메라를 에이전트의 자식으로 설정 (에이전트를 따라다님)
camera.position = (0,10,-8) # 에이전트 기준 카메라의 상대적 위치 (뒤쪽 위)
camera.rotation_x = 45        # 카메라의 초기 상하 각도 (아래를 보도록)
camera.rotation_y = 0         # 카메라의 초기 좌우 각도 (정면)
camera.fov = 75               # 카메라의 시야각 (Field of View)

# --- 유틸리티 함수 ---
def normalize_angle(rad):
    """ 라디안 각도를 -pi 에서 +pi 사이로 정규화합니다. """
    while rad > math.pi: rad -= 2*math.pi
    while rad < -math.pi: rad += 2*math.pi
    return rad

# --- 3D 라이다 시각화 함수 ---
lidar_lines = [] # Ursina 씬에 그려진 라이다 선들을 저장하는 리스트
def update_lidar_visualization():
    """ 현재 라이다 스캔 결과를 Ursina 씬에 실시간으로 그립니다. """
    global lidar_lines
    # 이전 프레임에서 그린 라인들을 제거
    for line in lidar_lines: destroy(line)
    lidar_lines.clear() # 리스트 비우기

    agent_pos=agent.world_position; agent_rot_y_deg=agent.world_rotation_y
    lidar_origin_world = agent_pos + Vec3(0,LIDAR_VIS_HEIGHT,0) # 라이다 센서의 월드 좌표
    
    # 수평/수직 각도 단계 계산
    h_angle_step=LIDAR_FOV/(NUM_LIDAR_RAYS-1) if NUM_LIDAR_RAYS>1 else 0
    v_angle_step=LIDAR_VERTICAL_FOV/(NUM_LIDAR_CHANNELS-1) if NUM_LIDAR_CHANNELS>1 else 0
    v_angle_start=-LIDAR_VERTICAL_FOV/2 # 수직 스캔 시작 각도 (중심 아래)

    # 모든 수직 채널(라인)에 대해 반복
    for ch in range(NUM_LIDAR_CHANNELS):
        v_angle_deg=v_angle_start+ch*v_angle_step; v_angle_rad=math.radians(v_angle_deg)
        cos_v=math.cos(v_angle_rad); sin_v=math.sin(v_angle_rad)
        h_angle_start_world_deg=agent_rot_y_deg-LIDAR_FOV/2 # 현재 채널의 수평 스캔 시작 각도 (월드 기준)
        
        # 모든 수평 광선(Ray)에 대해 반복
        for i in range(NUM_LIDAR_RAYS):
            h_angle_world_deg=h_angle_start_world_deg+i*h_angle_step; h_angle_world_rad=math.radians(h_angle_world_deg)
            cos_h=math.cos(h_angle_world_rad); sin_h=math.sin(h_angle_world_rad)
            direction=Vec3(cos_v*sin_h,sin_v,cos_v*cos_h).normalized() # 3D 광선 방향 벡터 (월드 좌표계)
            
            # 레이캐스팅 수행
            hit_info=raycast(lidar_origin_world,direction,LIDAR_RANGE,ignore=[agent,],debug=False,traverse_target=scene)
            
            if hit_info.hit: # 광선이 충돌한 경우
                end_point=hit_info.world_point
                is_obstacle=hasattr(hit_info.entity,'tag') and hit_info.entity.tag==OBSTACLE_TAG
                is_ground=hit_info.entity==ground
                line_color=color.red if is_obstacle else (color.gray if is_ground else color.orange)
            else: # 광선이 충돌하지 않은 경우
                end_point=lidar_origin_world+direction*LIDAR_RANGE; line_color=LIDAR_COLOR
            
            if distance(lidar_origin_world,end_point)>0.01: # 매우 짧은 선은 그리지 않음
                lidar_lines.append(Entity(model=Mesh(vertices=[lidar_origin_world,end_point],mode='line',thickness=1),color=line_color,alpha=0.7))

# --- 3D 라이다 스캔 및 상대 좌표 데이터 생성 함수 ---
def generate_relative_lidar_map_3d():
    """ 에이전트 현재 위치/자세 기준으로, 감지된 모든 충돌 포인트들의 상대 3D 좌표를 생성합니다. """
    relative_points_3d=[]
    agent_pos_world=agent.world_position; agent_rot_y_rad=math.radians(agent.world_rotation_y)
    scan_origin_world=agent_pos_world+Vec3(0,LIDAR_VIS_HEIGHT,0)
    h_angle_step=LIDAR_FOV/(NUM_LIDAR_RAYS-1) if NUM_LIDAR_RAYS>1 else 0
    v_angle_step=LIDAR_VERTICAL_FOV/(NUM_LIDAR_CHANNELS-1) if NUM_LIDAR_CHANNELS>1 else 0
    v_angle_start=-LIDAR_VERTICAL_FOV/2
    for ch in range(NUM_LIDAR_CHANNELS):
        v_angle_deg=v_angle_start+ch*v_angle_step; v_angle_rad=math.radians(v_angle_deg)
        cos_v=math.cos(v_angle_rad); sin_v=math.sin(v_angle_rad)
        h_angle_start_world_deg=agent.world_rotation_y-LIDAR_FOV/2
        for i in range(NUM_LIDAR_RAYS):
            h_angle_world_deg=h_angle_start_world_deg+i*h_angle_step; h_angle_world_rad=math.radians(h_angle_world_deg)
            cos_h=math.cos(h_angle_world_rad); sin_h=math.sin(h_angle_world_rad)
            direction=Vec3(cos_v*sin_h,sin_v,cos_v*cos_h).normalized()
            hit_info=raycast(scan_origin_world,direction,LIDAR_RANGE,ignore=[agent,],debug=False,traverse_target=scene)
            if hit_info.hit:
                hit_point_world=hit_info.world_point; world_vec=hit_point_world-agent_pos_world
                cos_a=math.cos(-agent_rot_y_rad); sin_a=math.sin(-agent_rot_y_rad)
                relative_x=world_vec.x*cos_a-world_vec.z*sin_a; relative_y=world_vec.y; relative_z=world_vec.x*sin_a+world_vec.z*cos_a
                relative_points_3d.append([relative_x,relative_y,relative_z])
    return np.array(relative_points_3d) if relative_points_3d else np.empty((0,3))

# --- 누적된 스캔 데이터로부터 전역 3D 맵 데이터 생성 함수 ---
def generate_map_data_3d(history):
    """ 스캔 기록(history)으로부터 전역 3D 맵 포인트 클라우드와 에이전트 궤적을 생성합니다. """
    global_map_points_list=[]; agent_trajectory_x,agent_trajectory_z=[],[]
    if not history: return np.empty((0,3)),[],[]
    for scan_record in history:
        pose=scan_record['pose']; relative_points=np.array(scan_record['relative_points'])
        if relative_points.shape[0]==0: continue
        scan_pos_x,scan_pos_z,scan_rot_rad=pose
        agent_pos_world_at_scan=Vec3(scan_pos_x,AGENT_HEIGHT,scan_pos_z) # 스캔 당시 에이전트 Y 위치 사용
        agent_trajectory_x.append(scan_pos_x); agent_trajectory_z.append(scan_pos_z)
        cos_a,sin_a=math.cos(scan_rot_rad),math.sin(scan_rot_rad)
        rot_matrix_2d=np.array([[cos_a,-sin_a],[sin_a,cos_a]])
        relative_xz=relative_points[:,[0,2]] # 상대 X, Z 좌표 추출
        rotated_xz=np.dot(relative_xz,rot_matrix_2d.T) # 2D 회전 변환
        world_x=rotated_xz[:,0]+scan_pos_x # 월드 X 좌표
        world_y=relative_points[:,1]+agent_pos_world_at_scan.y # 월드 Y 좌표 (상대 Y + 스캔 당시 에이전트 Y)
        world_z=rotated_xz[:,1]+scan_pos_z # 월드 Z 좌표
        world_points=np.stack((world_x,world_y,world_z),axis=-1) # (N,3) 형태로 결합
        global_map_points_list.append(world_points)
    map_points_np=np.concatenate(global_map_points_list,axis=0) if global_map_points_list else np.empty((0,3))
    return map_points_np,agent_trajectory_x,agent_trajectory_z

# --- 포인트 클라우드 전처리 함수: RANSAC 지면 분할 ---
def segment_ground_ransac(points_np, distance_threshold=0.2, ransac_n=3, num_iterations=100, visualize_segmentation=False):
    """
    RANSAC을 사용하여 포인트 클라우드에서 지면을 분할하고 비지면 포인트(장애물 후보)를 반환합니다.
    points_np: (N, 3) 형태의 NumPy 배열 (x, y(높이), z(깊이))
    """
    if points_np.shape[0]<ransac_n: return points_np,np.empty((0,3)) # 포인트 수 부족 시 원본 반환
    pcd=o3d.geometry.PointCloud(); pcd.points=o3d.utility.Vector3dVector(points_np) # Open3D 포인트 클라우드 객체 생성
    try:
        # RANSAC 평면 분할 실행
        plane_model,inliers_indices=pcd.segment_plane(distance_threshold,ransac_n,num_iterations)
        ground_pcd=pcd.select_by_index(inliers_indices) # 지면 포인트
        non_ground_pcd=pcd.select_by_index(inliers_indices,invert=True) # 비지면 포인트
        if visualize_segmentation: # 분할 결과 시각화 옵션
            ground_pcd.paint_uniform_color([0.5,0.5,0.5]); non_ground_pcd.paint_uniform_color([1,0,0]) # 회색/빨간색으로 표시
            o3d.visualization.draw_geometries([ground_pcd,non_ground_pcd],"RANSAC Segmentation",800,600)
        return np.asarray(non_ground_pcd.points),np.asarray(ground_pcd.points)
    except Exception as e: print(f"RANSAC error: {e}"); return points_np,np.empty((0,3))

# --- 3D 직육면체(Bounding Box) 그리기 헬퍼 함수 ---
def draw_cuboid(ax,min_coords,max_coords,color='r',alpha=0.1,linewidth=1,label=None):
    """ Matplotlib 3D 축(ax)에 주어진 최소/최대 좌표로 정의되는 직육면체를 그립니다.
        min_coords, max_coords는 (x, y(높이), z(깊이)) 순서로 가정합니다.
    """
    min_x,min_y_h,min_z_d=min_coords; max_x,max_y_h,max_z_d=max_coords
    # 직육면체의 8개 꼭지점 좌표 정의 (Matplotlib 3D 플롯은 X, Z(깊이), Y(높이) 순서로 축을 다룸)
    v=np.array([(min_x,min_z_d,min_y_h),(max_x,min_z_d,min_y_h),(max_x,max_z_d,min_y_h),(min_x,max_z_d,min_y_h),
                (min_x,min_z_d,max_y_h),(max_x,min_z_d,max_y_h),(max_x,max_z_d,max_y_h),(min_x,max_z_d,max_y_h)])
    # 직육면체의 12개 모서리를 정의하는 꼭지점 인덱스 쌍
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    plotted_label_flag=False # 라벨 중복 방지 플래그
    for i,edge in enumerate(edges):
        xs,zs_depth,ys_height=v[edge,0],v[edge,1],v[edge,2] # X, Z(깊이), Y(높이) 순서로 추출
        current_label=label if not plotted_label_flag else None # 첫 번째 모서리에만 라벨 적용
        ax.plot(xs,zs_depth,ys_height,color=color,alpha=alpha*2,linewidth=linewidth,label=current_label)
        if current_label: plotted_label_flag=True

# --- 3D IoU 계산 함수 ---
def calculate_3d_iou(box1_min_xyz,box1_max_xyz,box2_min_xyz,box2_max_xyz):
    """ 두 개의 축 정렬 3D 경계 상자 간의 IoU를 계산합니다.
        각 box_min/max_xyz는 (min_x, min_y(높이), min_z(깊이)) 형태의 튜플 또는 리스트입니다.
    """
    inter_min_x=max(box1_min_xyz[0],box2_min_xyz[0]); inter_min_y=max(box1_min_xyz[1],box2_min_xyz[1])
    inter_min_z=max(box1_min_xyz[2],box2_min_xyz[2]); inter_max_x=min(box1_max_xyz[0],box2_max_xyz[0])
    inter_max_y=min(box1_max_xyz[1],box2_max_xyz[1]); inter_max_z=min(box1_max_xyz[2],box2_max_xyz[2])
    inter_w=max(0,inter_max_x-inter_min_x); inter_h=max(0,inter_max_y-inter_min_y) # 교집합 높이
    inter_d=max(0,inter_max_z-inter_min_z); intersection_volume=inter_w*inter_h*inter_d # 교집합 부피
    vol1=(box1_max_xyz[0]-box1_min_xyz[0])*(box1_max_xyz[1]-box1_min_xyz[1])*(box1_max_xyz[2]-box1_min_xyz[2]) # 박스1 부피
    vol2=(box2_max_xyz[0]-box2_min_xyz[0])*(box2_max_xyz[1]-box2_min_xyz[1])*(box2_max_xyz[2]-box2_min_xyz[2]) # 박스2 부피
    union_volume=vol1+vol2-intersection_volume # 합집합 부피
    return intersection_volume/union_volume if union_volume > 0 else 0.0 # IoU 계산

# --- XZ 평면상 겹침 비율 계산 함수 ---
def calculate_2d_xz_overlap_ratio(pred_min_xz, pred_max_xz, gt_min_xz, gt_max_xz):
    """
    두 개의 XZ 평면상 AABB 간의 겹침 비율을 계산합니다.
    비율 = (교차 영역 넓이) / (GT 박스 넓이)
    pred_min_xz, pred_max_xz: 예측 박스의 (min_x, min_z), (max_x, max_z)
    gt_min_xz, gt_max_xz: GT 박스의 (min_x, min_z), (max_x, max_z)
    """
    # 교차 영역의 최소/최대 X, Z 좌표 계산
    inter_min_x = max(pred_min_xz[0], gt_min_xz[0]); inter_min_z = max(pred_min_xz[1], gt_min_xz[1])
    inter_max_x = min(pred_max_xz[0], gt_max_xz[0]); inter_max_z = min(pred_max_xz[1], gt_max_xz[1])
    # 교차 영역의 너비와 깊이 계산
    inter_w = max(0, inter_max_x - inter_min_x); inter_d = max(0, inter_max_z - inter_min_z)
    intersection_area = inter_w * inter_d
    # GT 박스의 XZ 평면 넓이 계산
    gt_area = (gt_max_xz[0] - gt_min_xz[0]) * (gt_max_xz[1] - gt_min_xz[1])
    return intersection_area / gt_area if gt_area > 0 else 0.0


# --- 통합 맵 플로팅 및 성능 평가 함수 ---
def plot_all_maps(history,full_gt_obstacle_df,gnd_info):
    """ 3개의 맵(포인트클라우드, GT맵, 탐지결과)을 그리고 성능을 평가하여 함께 표시합니다. """
    print("통합 맵 생성 및 성능 평가 중...")
    if not history: print("플롯할 스캔 기록이 없습니다."); return
    if full_gt_obstacle_df is None: print("Ground Truth 장애물 데이터가 없습니다."); return
    
    # 1. 누적된 스캔으로부터 전역 포인트 클라우드 및 에이전트 궤적 생성
    map_points_np_3d,agent_traj_x,agent_traj_z=generate_map_data_3d(history)
    print(f"생성된 총 전역 3D 포인트 개수: {len(map_points_np_3d)}")
    if len(map_points_np_3d)==0 and not agent_traj_x: print("생성된 포인트와 궤적 데이터가 모두 없습니다."); return
    
    # 2. LiDAR 범위 내에 있었던 GT 장애물 필터링
    relevant_gt_indices=set() 
    if not full_gt_obstacle_df.empty and history:
        print(f"LiDAR 범위 ({LIDAR_RANGE}m) 내 GT 장애물 필터링 중...")
        for gt_idx,gt_row in full_gt_obstacle_df.iterrows(): 
            gt_min_x=gt_row['center_x']-gt_row['width']/2; gt_max_x=gt_row['center_x']+gt_row['width']/2
            gt_min_y=gt_row['center_y']-gt_row['height']/2; gt_max_y=gt_row['center_y']+gt_row['height']/2
            gt_min_z=gt_row['center_z']-gt_row['depth']/2; gt_max_z=gt_row['center_z']+gt_row['depth']/2
            for scan_record in history:
                agent_scan_x,agent_scan_z,_=scan_record['pose'] 
                lidar_sensor_y=AGENT_HEIGHT+LIDAR_VIS_HEIGHT 
                lidar_sensor_center=Vec3(agent_scan_x,lidar_sensor_y,agent_scan_z) 
                closest_x=max(gt_min_x,min(lidar_sensor_center.x,gt_max_x))
                closest_y=max(gt_min_y,min(lidar_sensor_center.y,gt_max_y))
                closest_z=max(gt_min_z,min(lidar_sensor_center.z,gt_max_z))
                distance_sq=(closest_x-lidar_sensor_center.x)**2+(closest_y-lidar_sensor_center.y)**2+(closest_z-lidar_sensor_center.z)**2
                if distance_sq<=(LIDAR_RANGE**2): relevant_gt_indices.add(gt_idx); break 
    relevant_gt_obstacle_df=full_gt_obstacle_df.loc[list(relevant_gt_indices)].copy() if relevant_gt_indices else pd.DataFrame(columns=full_gt_obstacle_df.columns)
    print(f"전체 GT 장애물 수: {len(full_gt_obstacle_df)}, 범위 내 관련 GT 장애물 수: {len(relevant_gt_obstacle_df)}")

    # 3. 포인트 클라우드 전처리 (지면 제거 및 Y축 스케일링)
    non_ground_points_orig = map_points_np_3d 
    ground_points_ransac = np.empty((0,3))
    points_for_dbscan_scaled = np.empty((0,3)) 
    if len(map_points_np_3d) > 0:
        non_ground_points_orig_temp, ground_points_ransac_temp = segment_ground_ransac(map_points_np_3d,RANSAC_DISTANCE_THRESHOLD,RANSAC_N_POINTS,RANSAC_NUM_ITERATIONS,VISUALIZE_RANSAC_SEGMENTATION)
        if non_ground_points_orig_temp.shape[0] > 0 or ground_points_ransac_temp.shape[0] > 0 :
            non_ground_points_orig = non_ground_points_orig_temp
            ground_points_ransac = ground_points_ransac_temp
        if len(non_ground_points_orig) > 0:
            points_for_dbscan_scaled = non_ground_points_orig.copy() 
            points_for_dbscan_scaled[:, 1] *= Y_COORD_SCALE_FACTOR 
            print(f"Y축 스케일링 적용 (Factor: {Y_COORD_SCALE_FACTOR}). DBSCAN 입력 포인트 수: {len(points_for_dbscan_scaled)}")

    # 4. Matplotlib Figure 및 서브플롯 생성
    fig=plt.figure(figsize=(24,8.5)); fig.suptitle('3D LiDAR Mapping, Y-Scaled DBSCAN & Performance',fontsize=16) 
    
    # --- 서브플롯 1: 3D 라이다 포인트 클라우드 (전처리 결과 포함) ---
    ax1=fig.add_subplot(131,projection='3d')
    if len(map_points_np_3d)>0: ax1.scatter(map_points_np_3d[:,0],map_points_np_3d[:,2],map_points_np_3d[:,1],s=1,c=map_points_np_3d[:,1],cmap='Greys',alpha=0.1,label='Original')
    if len(ground_points_ransac)>0: ax1.scatter(ground_points_ransac[:,0],ground_points_ransac[:,2],ground_points_ransac[:,1],s=1,c='lightgray',alpha=0.2,label='Ground (RANSAC)')
    if len(non_ground_points_orig)>0: ax1.scatter(non_ground_points_orig[:,0],non_ground_points_orig[:,2],non_ground_points_orig[:,1],s=2,c=non_ground_points_orig[:,1],cmap='viridis',alpha=0.6,label='Non-Ground (Original)')
    if agent_traj_x:
        ax1.plot(agent_traj_x,agent_traj_z,zs=0,zdir='z',marker='.',linestyle='-',color='blue',label='Trajectory')
        ax1.scatter(agent_traj_x[-1],agent_traj_z[-1],zs=0,zdir='z',s=60,c='magenta',marker='*',label='Last Pose')
    ax1.set_title("Point Cloud (RANSAC Processed)"); ax1.set_xlabel("X"); ax1.set_ylabel("Z (Depth)"); ax1.set_zlabel("Y (Height)")
    limit=AREA_SIZE/1.8; ax1.set_xlim(-limit,limit); ax1.set_ylim(-limit,limit); ax1.set_zlim(0,10)
    ax1.view_init(elev=30.,azim=-60); ax1.legend(fontsize='small',markerscale=2)

    ax2=fig.add_subplot(132) # 2D GT 맵은 나중에 TP/FN/FP 상태와 함께 그림
    
    # --- 서브플롯 3: 탐지된 장애물 (3D Bounding Boxes) ---
    ax3=fig.add_subplot(133,projection='3d')
    detected_obstacle_count=0; predicted_boxes_info=[] 
    if len(points_for_dbscan_scaled) >= DBSCAN_MIN_SAMPLES: 
        try:
            print(f"DBSCAN 클러스터링 실행 중 (입력: Y-스케일된 포인트 {len(points_for_dbscan_scaled)}개)...")
            db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points_for_dbscan_scaled)
            labels = db.labels_
            unique_labels = set(labels)
            colors_map = plt.cm.get_cmap('Spectral', len(unique_labels))
            num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            print(f"DBSCAN: {num_clusters} clusters found.")
            cluster_label_printed = False
            for k_idx, k in enumerate(unique_labels):
                if k == -1: continue
                cluster_mask_for_bbox = (labels == k)
                original_cluster_points = non_ground_points_orig[cluster_mask_for_bbox] 
                if len(original_cluster_points) > 0:
                    min_c = np.min(original_cluster_points, axis=0); max_c = np.max(original_cluster_points, axis=0)
                    predicted_boxes_info.append({'min_coords': tuple(min_c), 'max_coords': tuple(max_c), 'is_tp': False}) 
                    lbl = 'Detected BBox' if not cluster_label_printed else None
                    draw_cuboid(ax3, min_c, max_c, color=colors_map(k_idx/len(unique_labels)), alpha=0.15, linewidth=1.5, label=lbl)
                    if lbl: cluster_label_printed = True
                    detected_obstacle_count += 1
        except Exception as e: print(f"DBSCAN/Plotting error: {e}"); ax3.text(0,0,0,'DBSCAN Error',c='red')
    else: print("Not enough points for Y-scaled DBSCAN."); ax3.text(0,0,0,'No Points for DBSCAN',c='gray')
    
    if agent_traj_x: 
        ax3.plot(agent_traj_x,agent_traj_z,zs=0,zdir='z',marker='.',linestyle='-',color='blue',label='Trajectory')
        ax3.scatter(agent_traj_x[-1],agent_traj_z[-1],zs=0,zdir='z',s=60,c='magenta',marker='*',label='Last Pose')
    
    ax3_title = f"Detected ({detected_obstacle_count} BBoxes)\n"
    ax3_title += f"DBSCAN: eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}, Y-Scale={Y_COORD_SCALE_FACTOR}"
    ax3.set_title(ax3_title, fontsize=10) 

    ax3.set_xlabel("X"); ax3.set_ylabel("Z (Depth)"); ax3.set_zlabel("Y (Height)")
    ax3.set_xlim(-limit,limit); ax3.set_ylim(-limit,limit); ax3.set_zlim(0,10)
    ax3.view_init(elev=30.,azim=-60)
    if detected_obstacle_count > 0 or agent_traj_x: ax3.legend(fontsize='small')
    
    # --- 성능 평가 로직 ---
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
            'is_detected': False, # 이 GT가 예측에 의해 "탐지" (커버)되었는지 여부
            'original_gt_idx':idx
        })

    tp = 0; fp = 0 # TP, FP는 예측(predicted_boxes_info) 기준으로 카운트

    if EVALUATION_METHOD == "3D_IOU":
        print("평가 방식: 3D IoU (예측 중심)")
        # 각 GT가 어떤 예측과 매칭되었는지 추적 (가장 높은 IoU로 매칭된 예측의 인덱스)
        temp_gt_matched_by_pred_idx = [-1] * len(gt_boxes_info_for_eval)
        # 각 예측이 GT와 매칭될 때의 IoU 값 저장 (더 좋은 매칭으로 갱신하기 위함)
        pred_box_iou_with_gt = [0.0] * len(predicted_boxes_info)

        for pred_idx, pred_box_info in enumerate(predicted_boxes_info):
            best_iou_for_this_pred = 0
            best_gt_eval_idx_for_this_pred = -1
            for gt_eval_idx, gt_box_eval_info in enumerate(gt_boxes_info_for_eval):
                iou = calculate_3d_iou(pred_box_info['min_coords'], pred_box_info['max_coords'], 
                                       gt_box_eval_info['min_coords_3d'], gt_box_eval_info['max_coords_3d'])
                if iou > best_iou_for_this_pred:
                    best_iou_for_this_pred = iou
                    best_gt_eval_idx_for_this_pred = gt_eval_idx
            
            if best_iou_for_this_pred >= IOU_THRESHOLD and best_gt_eval_idx_for_this_pred != -1:
                gt_idx_to_match = best_gt_eval_idx_for_this_pred
                # 현재 예측이 이 GT와 매칭될 수 있는지 확인
                # 1. 이 GT가 아직 다른 예측과 매칭되지 않았거나
                # 2. 또는 이 예측이 이 GT와 이전 매칭보다 더 높은 IoU를 가질 경우
                if temp_gt_matched_by_pred_idx[gt_idx_to_match] == -1 or \
                   best_iou_for_this_pred > pred_box_iou_with_gt[temp_gt_matched_by_pred_idx[gt_idx_to_match]]:
                    
                    # 만약 이 GT가 이전에 다른 예측과 매칭되었다면, 그 이전 예측은 이제 TP가 아님 (is_tp = False)
                    if temp_gt_matched_by_pred_idx[gt_idx_to_match] != -1:
                        predicted_boxes_info[temp_gt_matched_by_pred_idx[gt_idx_to_match]]['is_tp'] = False
                    
                    predicted_boxes_info[pred_idx]['is_tp'] = True # 현재 예측을 TP로 설정
                    pred_box_iou_with_gt[pred_idx] = best_iou_for_this_pred # 현재 예측의 IoU 저장
                    gt_boxes_info_for_eval[gt_idx_to_match]['is_detected'] = True # 이 GT는 탐지됨
                    temp_gt_matched_by_pred_idx[gt_idx_to_match] = pred_idx # 이 GT는 현재 예측과 매칭됨
        
        tp = sum(1 for p_box in predicted_boxes_info if p_box['is_tp'])
        fp = len(predicted_boxes_info) - tp

    elif EVALUATION_METHOD == "2D_CENTER_IN_GT_AREA":
        print("평가 방식: 2D Center-in-GT-Area (예측 중심)")
        for pred_idx, pred_box_info in enumerate(predicted_boxes_info):
            pred_center_x = (pred_box_info['min_coords'][0] + pred_box_info['max_coords'][0]) / 2
            pred_center_z = (pred_box_info['min_coords'][2] + pred_box_info['max_coords'][2]) / 2
            for gt_eval_idx, gt_box_eval_info in enumerate(gt_boxes_info_for_eval):
                gt_min_x, gt_min_z = gt_box_eval_info['min_coords_2d_xz']
                gt_max_x, gt_max_z = gt_box_eval_info['max_coords_2d_xz']
                if (gt_min_x <= pred_center_x <= gt_max_x) and (gt_min_z <= pred_center_z <= gt_max_z):
                    if not predicted_boxes_info[pred_idx]['is_tp']: 
                        predicted_boxes_info[pred_idx]['is_tp'] = True
                        # tp +=1 # tp는 아래에서 is_tp 기준으로 한번만 계산
                    gt_boxes_info_for_eval[gt_eval_idx]['is_detected'] = True 
                    # break # 한 예측은 하나의 GT에만 TP로 기여하도록 하려면 break. 여기서는 is_detected만 업데이트.
        tp = sum(1 for p_box in predicted_boxes_info if p_box['is_tp'])
        fp = len(predicted_boxes_info) - tp # is_tp가 아닌 모든 예측은 FP

    elif EVALUATION_METHOD == "2D_AREA_OVERLAP_RATIO":
        print(f"평가 방식: 2D XZ Area Overlap Ratio (GT 기준, 임계값: {AREA_OVERLAP_RATIO_THRESHOLD})")
        # 각 GT가 어떤 예측과 매칭되었는지 기록 (중복 TP 방지용)
        gt_best_pred_match_idx = [-1] * len(gt_boxes_info_for_eval) # 각 GT와 가장 잘 맞는 예측의 인덱스
        gt_best_pred_overlap = [0.0] * len(gt_boxes_info_for_eval)   # 그때의 overlap ratio

        for pred_idx, pred_box_info in enumerate(predicted_boxes_info):
            pred_min_xz = (pred_box_info['min_coords'][0], pred_box_info['min_coords'][2])
            pred_max_xz = (pred_box_info['max_coords'][0], pred_box_info['max_coords'][2])
            
            for gt_eval_idx, gt_box_eval_info in enumerate(gt_boxes_info_for_eval):
                overlap_ratio = calculate_2d_xz_overlap_ratio(
                    pred_min_xz, pred_max_xz,
                    gt_box_eval_info['min_coords_2d_xz'], gt_box_eval_info['max_coords_2d_xz']
                )
                if overlap_ratio >= AREA_OVERLAP_RATIO_THRESHOLD:
                    # 이 예측은 이 GT와 충분히 겹침
                    gt_boxes_info_for_eval[gt_eval_idx]['is_detected'] = True # 이 GT는 어쨌든 탐지됨 (FN 계산용)
                    
                    # 이 예측이 이 GT와 "가장 잘" 맞는 예측인지 확인
                    if overlap_ratio > gt_best_pred_overlap[gt_eval_idx]:
                        # 만약 이 GT가 이전에 다른 예측과 연결되어 있었다면, 그 이전 예측은 is_tp=False로 되돌릴 필요는 없음
                        # (왜냐하면 예측은 여러 GT와 겹칠 수 있고, 그 중 하나라도 TP 조건 만족하면 TP)
                        # 여기서는 각 GT에 대해 가장 overlap이 좋은 예측을 기억해둠
                        gt_best_pred_overlap[gt_eval_idx] = overlap_ratio
                        gt_best_pred_match_idx[gt_eval_idx] = pred_idx
        
        # TP는 예측 기준으로, 각 예측이 어떤 GT와든 기준 이상 겹치면 TP
        # 그리고 하나의 GT는 하나의 예측에 의해서만 TP로 카운트되도록 함 (위에서 gt_best_pred_match_idx 사용)
        # 위 로직 수정: 예측이 어떤 GT와든 기준 이상 겹치면 is_tp = True로 먼저 설정
        for pred_idx, pred_box_info in enumerate(predicted_boxes_info):
            pred_min_xz = (pred_box_info['min_coords'][0], pred_box_info['min_coords'][2])
            pred_max_xz = (pred_box_info['max_coords'][0], pred_box_info['max_coords'][2])
            for gt_eval_idx, gt_box_eval_info in enumerate(gt_boxes_info_for_eval):
                 overlap_ratio = calculate_2d_xz_overlap_ratio(
                    pred_min_xz, pred_max_xz,
                    gt_box_eval_info['min_coords_2d_xz'], gt_box_eval_info['max_coords_2d_xz']
                )
                 if overlap_ratio >= AREA_OVERLAP_RATIO_THRESHOLD:
                     predicted_boxes_info[pred_idx]['is_tp'] = True # 일단 TP 후보로 설정
                     # gt_boxes_info_for_eval[gt_eval_idx]['is_detected'] = True # 이미 위에서 처리됨
                     break # 이 예측은 하나 이상의 GT와 충분히 겹치므로 더 볼 필요 없음
        
        tp = sum(1 for p_box in predicted_boxes_info if p_box['is_tp'])
        fp = len(predicted_boxes_info) - tp
    else: print(f"알 수 없는 평가 방식: {EVALUATION_METHOD}")

    fn = sum(1 for gt_box_eval_info in gt_boxes_info_for_eval if not gt_box_eval_info['is_detected'])
    
    # --- 2D GT 맵에 TP/FN/FP 시각화 ---
    gnd_size=gnd_info['size']; ax2.add_patch(patches.Rectangle((-gnd_size/2,-gnd_size/2),gnd_size,gnd_size,ec='gray',fc='none',ls='--'))
    tp_label_added=False; fn_label_added=False; other_gt_label_added = False; fp_label_added = False
    
    # 1. GT 장애물 (TP, FN, Other) 그리기
    for idx, row in full_gt_obstacle_df.iterrows(): 
        current_label = None; rect_color = 'lightgrey'; rect_alpha = 0.3 
        is_relevant_gt = False
        gt_is_detected_for_plot = False 
        for gt_eval_info in gt_boxes_info_for_eval:
            if gt_eval_info['original_gt_idx'] == idx:
                is_relevant_gt = True
                gt_is_detected_for_plot = gt_eval_info['is_detected']
                break
        if is_relevant_gt:
            if gt_is_detected_for_plot: 
                rect_color = 'forestgreen'; rect_alpha = 0.7 
                if not tp_label_added: current_label = 'TP GT (Detected)'; tp_label_added = True
            else: 
                rect_color = 'crimson'; rect_alpha = 0.6 
                if not fn_label_added: current_label = 'FN GT (Missed)'; fn_label_added = True
        elif not other_gt_label_added: 
             current_label = 'Other GT (Out of Range)'; other_gt_label_added = True
        ax2.add_patch(patches.Rectangle((row['center_x']-row['width']/2,row['center_z']-row['depth']/2),
                                        row['width'],row['depth'],ec='k',fc=rect_color,alpha=rect_alpha,label=current_label))
    
    # 2. FP 예측 그리기
    for pred_box_info in predicted_boxes_info:
        if not pred_box_info['is_tp']: # TP가 아닌 예측은 FP
            pred_min_x = pred_box_info['min_coords'][0]
            pred_max_x = pred_box_info['max_coords'][0]
            pred_min_z = pred_box_info['min_coords'][2]
            pred_max_z = pred_box_info['max_coords'][2]
            fp_width = pred_max_x - pred_min_x
            fp_depth = pred_max_z - pred_min_z
            current_label = None
            if not fp_label_added:
                current_label = 'FP Prediction'
                fp_label_added = True
            ax2.add_patch(patches.Rectangle((pred_min_x, pred_min_z), fp_width, fp_depth, 
                                            edgecolor='darkorange', facecolor='none', hatch='//', alpha=0.7, linestyle='--', label=current_label)) # facecolor='none', hatch='//' 로 변경

    ax2.scatter(0,0,s=100,c='red',marker='x',label='Start'); ax2.set_title("Ground Truth Map (2D - TP/FN/FP Status)")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Z (m)"); ax2.grid(True,ls='--',alpha=0.6); ax2.set_aspect('equal','box')
    ax2.legend(fontsize='small'); ax2.set_xlim(-limit,limit); ax2.set_ylim(-limit,limit)

    plt.tight_layout(rect=[0, 0.05, 1, 0.92]) 

    precision=tp/(tp+fp) if (tp+fp)>0 else 0; recall=tp/(tp+fn) if (tp+fn)>0 else 0
    f1_score=2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0
    
    eval_method_str = EVALUATION_METHOD.replace("_", " ")
    threshold_str = ""
    if EVALUATION_METHOD == "3D_IOU": threshold_str = f" (IoU Thresh: {IOU_THRESHOLD})"
    elif EVALUATION_METHOD == "2D_AREA_OVERLAP_RATIO": threshold_str = f" (Overlap Thresh: {AREA_OVERLAP_RATIO_THRESHOLD})"
    
    eval_text = f"Performance (Eval: {eval_method_str}{threshold_str}, LiDAR Range:{LIDAR_RANGE}m)\n"
    eval_text += f"  Relevant GT Obstacles: {len(gt_boxes_info_for_eval)}\n  Predicted BBoxes: {len(predicted_boxes_info)}\n"
    eval_text += f"  TP: {tp}, FP: {fp}, FN: {fn}\n  Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1_score:.3f}"
    print(eval_text)
    fig.text(0.5, 0.02, eval_text, ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
    plt.show(block=True)

info_text = Text(origin=(-0.5,0.5), scale=(0.8,0.8), x=-0.5*window.aspect_ratio+0.02, y=0.48, text="Initializing...")

def input(key):
    global scan_history,obstacle_df,movement_states,rotation_states
    if key in ['w','w hold']: movement_states['forward']=True
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
    elif key in ['m','M']: plot_all_maps(scan_history,obstacle_df,ground_info)
    elif key in ['c','C']: scan_history=[]; print("Scan history cleared.")
    elif key in ['p','P']:
        ts=time.strftime('%Y%m%d_%H%M%S'); lr_str=str(LIDAR_RANGE).replace('.','_'); ys_str=str(Y_COORD_SCALE_FACTOR).replace('.','_')
        eps_str=str(DBSCAN_EPS).replace('.','_'); min_s_str=str(DBSCAN_MIN_SAMPLES)
        eval_m_str = EVALUATION_METHOD.lower().replace("3d_iou","iou").replace("2d_center_in_gt_area","center").replace("2d_area_overlap_ratio","area")
        if scan_history: fn_hist=f"scan_s{RANDOM_OBSTACLE_SEED}_lr{lr_str}_ys{ys_str}_eps{eps_str}_ms{min_s_str}_eval{eval_m_str}_{ts}.pkl"; pickle.dump(scan_history,open(fn_hist,'wb')); print(f"Hist saved: {fn_hist}")
        if obstacle_df is not None: fn_obs=f"gt_s{RANDOM_OBSTACLE_SEED}_lr{lr_str}_ys{ys_str}_eps{eps_str}_ms{min_s_str}_eval{eval_m_str}_{ts}.csv"; obstacle_df.to_csv(fn_obs,index=False); print(f"GT saved: {fn_obs}")
    elif key=='escape': application.quit()

def update():
    global scan_timer,scan_history,movement_states,rotation_states
    original_pos=agent.position; total_dx,total_dz=0.0,0.0; speed_dt=AGENT_SPEED*time.dt
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
    scan_timer+=time.dt
    if scan_timer>=SCAN_INTERVAL:
        scan_timer-=SCAN_INTERVAL; current_pos_xz=agent.world_position.xz; current_rot_rad=math.radians(agent.world_rotation_y)
        current_pose=(current_pos_xz.x,current_pos_xz.y,current_rot_rad)
        rel_pts=generate_relative_lidar_map_3d()
        if rel_pts.shape[0]>0: scan_history.append({'pose':current_pose,'relative_points':rel_pts.tolist()})
    
    if VISUALIZE_LIDAR_RAYS: 
        update_lidar_visualization()
    else: 
        global lidar_lines
        for line in lidar_lines: destroy(line)
        lidar_lines.clear()

    info_text.text=f"Pos:({agent.x:.1f},{agent.z:.1f}) RotY:{agent.rotation_y:.0f}° Scans:{len(scan_history)}\nRange:{LIDAR_RANGE}m YScl:{Y_COORD_SCALE_FACTOR} Eps:{DBSCAN_EPS} MinS:{DBSCAN_MIN_SAMPLES}\nEval: {EVALUATION_METHOD.replace('_',' ')}\nM:Map C:Clr P:Save Seed:{RANDOM_OBSTACLE_SEED}"
    info_text.x=-0.5*window.aspect_ratio+0.02; info_text.y=0.45 
    if mouse.locked:mouse.locked=False
    if not mouse.visible:mouse.visible=True

if __name__ == '__main__':
    app.run()
