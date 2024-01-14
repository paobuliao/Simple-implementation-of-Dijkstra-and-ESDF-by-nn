import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import csv
import os
from queue import PriorityQueue
from scipy.optimize import minimize
import time


def generate_random_map(size=64, num_obstacles=10):
    # Generate a random map with obstacles
    random_map = np.zeros((size, size), dtype=int)

    for _ in range(num_obstacles):
        obstacle_size = np.random.randint(3, 10)
        obstacle_type = np.random.choice([1, 2])

        if obstacle_type == 1:
            x, y = np.random.randint(0, size - obstacle_size, 2)
            random_map[x:x+obstacle_size, y:y+obstacle_size] = 1
        else:
            center = np.random.randint(obstacle_size, size - obstacle_size, 2)
            y, x = np.ogrid[:size, :size]
            mask = ((x - center[0])**2 + (y - center[1])**2) < obstacle_size**2
            random_map[mask] = 1
    flata_map=random_map.copy()
    #膨胀1
    for i in range(1,random_map.shape[0]-1):
        for j in range(1,random_map.shape[1]-1):
            if random_map[i,j]==1:
                flata_map[i-1,j-1]=1
                flata_map[i-1,j]=1
                flata_map[i-1,j+1]=1
                flata_map[i,j-1]=1
                flata_map[i,j+1]=1
                flata_map[i+1,j-1]=1
                flata_map[i+1,j]=1
                flata_map[i+1,j+1]=1
            
    return random_map,flata_map

def visualize_map(map_array):
    # Visualize the map
    plt.imshow(map_array, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('Random Map')
    plt.show()

def heuristic(a, b):
    # A* heuristic (Euclidean distance)
    return np.linalg.norm(np.array(a) - np.array(b))

def astar(start, goal, random_map):
    # A* path planning
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set()

    while not queue.empty():
        cost, current = queue.get()

        if current == goal:
            return reconstruct_path(start, goal, came_from)

        if current in visited:
            continue

        visited.add(current)

        for next_step in get_neighbors(current, random_map.shape):
            new_cost = cost + 1  # Assuming a cost of 1 for each step
            if next_step not in visited and random_map[next_step[0], next_step[1]] == 0:
                priority = new_cost + heuristic(goal, next_step)
                queue.put((priority, next_step))
                came_from[next_step] = current

    return None

def get_neighbors(position, shape):
    # Get valid neighbors (within bounds)
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            new_position = (position[0] + i, position[1] + j)
            if 0 <= new_position[0] < shape[0] and 0 <= new_position[1] < shape[1]:
                neighbors.append(new_position)
    return neighbors

def reconstruct_path(start, goal, came_from):
    # Reconstruct the path from start to goal
    path = [goal]
    current = goal
    while current != start:
        current = came_from[current]
        path.append(current)
    return np.array(path)

def visualize_astar(random_map, path, start, goal,ori_path=None):
    # Visualize the A* result
    plt.imshow(random_map, cmap='gray', interpolation='nearest')
    
    if ori_path is not None:
        plt.plot(ori_path[:, 1], ori_path[:, 0], '-g', linewidth=2)
    plt.plot(path[:, 1], path[:, 0], '-r', linewidth=2)
    plt.scatter(start[1], start[0], color='green', marker='o')
    plt.scatter(goal[1], goal[0], color='blue', marker='o')
    plt.legend()
    plt.title('Path Searching')
    plt.show()

def save_data_csv(map_idx, random_map, start, goal, path, csv_writer):
    # Append the map, start, goal, and path to the CSV file
    csv_writer.writerow([f"Map_{map_idx}"])

    temp_map=random_map.copy()
    #temp_mao中start,goal,path的值为2,3,4
    temp_map[start[0],start[1]]=2
    temp_map[goal[0],goal[1]]=3
    #保存temp_map,csv中的一行
    csv_writer.writerow([f"Mapsg_{map_idx}"])
    for i in range(temp_map.shape[0]):
        csv_writer.writerow(temp_map[i])

    path_map=np.zeros_like(random_map)
    #将轨迹点取整，要在地图范围内
    x=path[:,0].astype(np.int32)
    y=path[:,1].astype(np.int32)
    for i in range(path.shape[0]):
        path_map[x,y]=1
    #先写入一行名称
    csv_writer.writerow([f"Path_{map_idx}"])
    for i in range(path_map.shape[0]):
        csv_writer.writerow(path_map[i])

def generate_and_save_maps_csv(num_maps, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        for map_idx in range(num_maps):
            print(f"Generating map {map_idx}...")
            # Generate random map
            random_map,flate_map = generate_random_map()

            # Set valid start and goal points
            start_point, goal_point = generate_valid_start_goal(random_map)

            # Run A* algorithm
            came_from = {}
            path = astar(tuple(start_point), tuple(goal_point), random_map)

            # Save data if A* found a path
            if path is not None:
                path = np.array(path)
                save_data_csv(map_idx, random_map, start_point, goal_point, path, csv_writer)
                visualize_astar(random_map, path, start_point, goal_point)
            else:
                print(f"A* did not find a path for map {map_idx}.")

def generate_valid_start_goal(random_map):
    # Generate valid start and goal points that are not in obstacle regions
    while True:
        start_point = np.random.randint(0, random_map.shape[0], 2)
        goal_point = np.random.randint(0, random_map.shape[1], 2)

        if random_map[start_point[0], start_point[1]] == 0 and random_map[goal_point[0], goal_point[1]] == 0:
            return start_point, goal_point


def dijkstra(start, goal, random_map):
    # Dijkstra's path planning
    queue = PriorityQueue()
    queue.put((0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while not queue.empty():
        cost, current = queue.get()

        if current == goal:
            return reconstruct_path(start, goal, came_from)#, cost_so_far[goal]

        # Skip if the current node has already been visited with a lower cost
        if cost > cost_so_far[current]:
            continue

        for next_step in get_neighbors(current, random_map.shape):
            #判断是否为障碍物
            if random_map[next_step[0], next_step[1]] == 1:
                continue
            #斜对角线cost为1.414
            new_cost = cost + 1 if abs(next_step[0]-current[0]) + abs(next_step[1]-current[1])<=1 else cost + 1.414
            if next_step not in cost_so_far or new_cost < cost_so_far[next_step]:
                cost_so_far[next_step] = new_cost
                priority = new_cost
                queue.put((priority, next_step))
                came_from[next_step] = current

    return None  # No path found

def minimal_snap(path, order=7):
    """
    Generate a minimum snap trajectory for the given path.

    Parameters:
    - path: 2D array representing the waypoints [(x1, y1), (x2, y2), ...]
    - order: Order of the polynomial (default is 7 for a cubic polynomial)

    Returns:
    - traj_points: 2D array representing the trajectory points [(x1, y1), (x2, y2), ...]
    """
    num_waypoints = len(path)
    num_coefficients = order + 1  # Number of coefficients in the polynomial
    if num_waypoints < 3:
        return None
    # Objective function to minimize (sum of squared accelerations)
    def objective(coefficients):
        accelerations = np.zeros(num_waypoints - 2)
        for i in range(num_waypoints - 2):
            accelerations[i] = 2 * coefficients[i * num_coefficients + 6]
        return np.sum(accelerations ** 2)

    # Constraint: Ensure the trajectory passes through each waypoint
    def constraint(coefficients):
        constraints = np.zeros(num_waypoints * 2)
        for i in range(num_waypoints):
            constraints[i * 2] = np.polyval(coefficients[i * num_coefficients:(i + 1) * num_coefficients], path[i][0]) - path[i][0]
            constraints[i * 2 + 1] = np.polyval(coefficients[i * num_coefficients:(i + 1) * num_coefficients], path[i][1]) - path[i][1]
        return constraints

    # Initial guess for the coefficients (all zeros)
    initial_guess = np.zeros(num_waypoints * num_coefficients)

    # Minimize the objective function subject to the constraint
    result = minimize(objective, initial_guess, constraints={'type': 'eq', 'fun': constraint})

    # Extract the optimized coefficients
    optimized_coefficients = result.x

    #根据优化后的系数，生成轨迹
    traj_points = []
    for i in range(num_waypoints):
        traj_points.append((np.polyval(optimized_coefficients[i * num_coefficients:(i + 1) * num_coefficients], path[i][0]),
                            np.polyval(optimized_coefficients[i * num_coefficients:(i + 1) * num_coefficients], path[i][1])))
    return np.array(traj_points)

#该函数检查一条轨迹有没有经过障碍物
def check_path(path,random_map):
    for i in range(path.shape[0]):
        #path点变为整数
        x=int(path[i][0])
        y=int(path[i][1])
        if random_map[x,y]==1:
            return False
    return True

def generate_and_save_maps_csv_dijkstra(num_maps, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        for map_idx in range(num_maps):
            print(f"Generating map {map_idx}...")
            # Generate random map
            random_map,flate_map = generate_random_map()

            
            # Set valid start and goal points
            start_point, goal_point = generate_valid_start_goal(flate_map)

            # Run Dijkstra's algorithm
            came_from.clear()
            
            start_time=time.time()
            
            path= dijkstra(tuple(start_point), tuple(goal_point), flate_map)
            end_time=time.time()
            print(f"cost time:{end_time-start_time}")
            
            # Save data if Dijkstra found a path
            if path is not None:
                path = np.array(path)
                #对path上的点，进行抽样，间隔为3
                inter_path=path[::7]
                
                # Smooth the path using minimal snap
                # smooth_path=minimal_snap(inter_path,7)
                # if smooth_path is None:
                #     save_data_csv(map_idx, random_map, start_point, goal_point, path, csv_writer)
                #     continue
                # if not check_path(smooth_path,flate_map):
                #     smooth_path=minimal_snap(path,7)
                #     save_data_csv(map_idx, random_map, start_point, goal_point, path, csv_writer)
                save_data_csv(map_idx, random_map, start_point, goal_point, path, csv_writer)
                # visualize_astar(random_map, path,start_point, goal_point)
            else:
                print(f"Dijkstra did not find a path for map {map_idx}.")


came_from={}
random_map = generate_random_map()
# Specify the CSV file path

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
csv_file_path_dijkstra = os.path.join(parent_directory, 'all_maps_and_paths.csv')

# Generate and save multiple maps and paths in the same CSV file
num_maps_to_generate = 1000
# generate_and_save_maps_csv(num_maps_to_generate, csv_file_path)
generate_and_save_maps_csv_dijkstra(num_maps_to_generate, csv_file_path_dijkstra)