import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import time
import heapq
from warehouse_env import WarehouseEnv

# Directions and their respective orientation codes
DIRECTIONS = {
    0: (-1, 0),  # Up
    1: (0, 1),   # Right
    2: (1, 0),   # Down
    3: (0, -1),  # Left
}

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def valid_move(grid, x, y):
    return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != 1

def a_star(grid, start, goal):
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}

    while open_list:
        _, g, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in DIRECTIONS.values():
            next_pos = (current[0] + dx, current[1] + dy)

            if not valid_move(grid, next_pos[0], next_pos[1]):
                continue
            tentative_g = g + 1
            if next_pos not in g_score or tentative_g < g_score[next_pos]:
                g_score[next_pos] = tentative_g
                f_score = tentative_g + heuristic(next_pos, goal)
                heapq.heappush(open_list, (f_score, tentative_g, next_pos))
                came_from[next_pos] = current
    return None
# Convert orientation steps to action list
def convert_path_to_actions(path, start_orientation):
    actions = []
    current_orient = start_orientation

    for i in range(1, len(path)):
        prev = path[i - 1]
        curr = path[i]
        dx, dy = curr[0] - prev[0], curr[1] - prev[1]

        # Determine desired direction
        for orient, (odx, ody) in DIRECTIONS.items():
            if (dx, dy) == (odx, ody):
                desired_orient = orient
                break

        # Calculate minimal rotation
        rot = (desired_orient - current_orient) % 4
        if rot == 1:
            actions.append(3)  # rotate right
        elif rot == 3:
            actions.append(2)  # rotate left
        elif rot == 2:
            actions.append(3)  # right
            actions.append(3)  # right

        # Move forward
        actions.append(0)
        current_orient = desired_orient
    actions.pop()
    return actions
def find_nearest_free_agent(agents_info, target_pos, agent_busy):
    min_dist = float('inf')
    nearest_idx = None
    for idx, agent in enumerate(agents_info):
        if agent_busy[idx]:  # skip busy agents
            continue
        agent_pos = tuple(agent['position'])
        dist = abs(agent_pos[0] - target_pos[0]) + abs(agent_pos[1] - target_pos[1])
        if dist < min_dist:
            min_dist = dist
            nearest_idx = idx
    return nearest_idx

env = WarehouseEnv(grid_size=(15, 10), num_agents=2)
obs = env.reset()

agent_plans = {i: [] for i in range(env.num_agents)}
agent_busy = {i: False for i in range(env.num_agents)}

timestep = 0
while env.step([6] * env.num_agents)[4]["requests"]:
    actions = [6] * env.num_agents  # idle by default

    obs, reward, done, truncated, info = env.step(actions)
    agents_info = info['agents']
    requests_info = info['requests']

    # Assign new requests only to free agents
    for req_idx, req in enumerate(requests_info):
        if req['picked'] == 0:
            source_pos = tuple(req['source'])
            nearest_agent = find_nearest_free_agent(agents_info, source_pos,agent_busy)

            if nearest_agent is not None and not agent_plans[nearest_agent]:
                agent_pos = tuple(agents_info[nearest_agent]['position'])
                agent_orient = agents_info[nearest_agent]['orientation']
                req["picked"] = 1

                # 1️⃣ Path to pickup
                path_to_pickup = a_star(env.grid, agent_pos, source_pos)
                if path_to_pickup:
                    actions_to_pickup = convert_path_to_actions(path_to_pickup,agent_orient)
                    actions_to_pickup.append(4)  # add pickup action
            
                    # 2️⃣ Path to drop
                    current_pos = path_to_pickup[-1]
                    if len(path_to_pickup) >= 2:
                        prev_pos = path_to_pickup[-2]
                        dx, dy = current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1]
                        for orient, (odx, ody) in DIRECTIONS.items():
                            if (dx, dy) == (odx, ody):
                                current_orient = orient
                                break
                            else:
                                current_orient = agent_orient  # no movement
                    drop_pos = tuple(req['destination'])
                    path_to_drop = a_star(env.grid, current_pos, drop_pos)
                    print("drop loaction",drop_pos)
                    if path_to_drop:
                        actions_to_drop = convert_path_to_actions(path_to_drop, current_orient)
                        actions_to_drop.append(5)  # add place action
                        # Combine full plan
                        agent_plans[nearest_agent] = actions_to_pickup + actions_to_drop
                        agent_busy[nearest_agent] = True
    # Execute one action per agent
    for i in range(env.num_agents):
        if agent_plans[i]:
            actions[i] = agent_plans[i].pop(0)
            if actions[i] == 5:
                agent_busy[i] = False
        else:
            actions[i] = 6  # idle
    #print(agent_plans)
    env.render()
    #print(actions)
    obs, reward, done, truncated, info = env.step(actions)
    
    timestep += 1
    if timestep > 1000:
        break
