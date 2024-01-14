import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import time
import csv
import os
def generate_esdf(grid_map, resolution,enforce_obstable=False):
    """
    Generate Euclidean Signed Distance Field (ESDF) from a 2D grid map.

    Parameters:
    - grid_map: 2D binary grid map (1: occupied, 0: free)
    - resolution: Resolution of the grid map

    Returns:
    - esdf: Euclidean Signed Distance Field
    """
    # Convert grid map to 3D voxel map
    voxel_map = np.zeros_like(grid_map, dtype=float)
    voxel_map[grid_map == 1] = np.inf

    # Compute Euclidean distance transform
    distance_transform = distance_transform_edt(grid_map, sampling=[resolution, resolution])

    # Invert the distance transform for free space
    esdf = -distance_transform
    if enforce_obstable:
        for i in range(1, esdf.shape[0]-1):
            for j in range(1, esdf.shape[1]-1):
                if esdf[i, j] == 0:
                    esdf[i,j]=100
    return esdf

def visualize_esdf(esdf, resolution):
    """
    Visualize Euclidean Signed Distance Field (ESDF).

    Parameters:
    - esdf: Euclidean Signed Distance Field
    - resolution: Resolution of the ESDF
    """
    plt.imshow(esdf, extent=(0, esdf.shape[1] * resolution, 0, esdf.shape[0] * resolution))
    plt.colorbar(label='Signed Distance (meters)')
    plt.title('Euclidean Signed Distance Field (ESDF)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.show()

def generate_random_map(size=64, num_obstacles=10):
    # Generate a random map with obstacles
    random_map = np.ones((size, size), dtype=int)

    for _ in range(num_obstacles):
        obstacle_size = np.random.randint(3, 10)
        obstacle_type = np.random.choice([1, 2])

        if obstacle_type == 1:
            x, y = np.random.randint(0, size - obstacle_size, 2)
            random_map[x:x+obstacle_size, y:y+obstacle_size] = 0
        else:
            center = np.random.randint(obstacle_size, size - obstacle_size, 2)
            y, x = np.ogrid[:size, :size]
            mask = ((x - center[0])**2 + (y - center[1])**2) < obstacle_size**2
            random_map[mask] = 0

    flata_map = random_map.copy()

    # 膨胀1
    for i in range(1, random_map.shape[0]-1):
        for j in range(1, random_map.shape[1]-1):
            if random_map[i, j] == 1:
                flata_map[i-1, j-1] = 1
                flata_map[i-1, j] = 1
                flata_map[i-1, j+1] = 1
                flata_map[i, j-1] = 1
                flata_map[i, j+1] = 1
                flata_map[i+1, j-1] = 1
                flata_map[i+1, j] = 1
                flata_map[i+1, j+1] = 1

    return random_map, flata_map

def generate_esdf_from_random_map(random_map, resolution, enforce_obstable=False):
    """
    Generate Euclidean Signed Distance Field (ESDF) from a random map.

    Parameters:
    - random_map: 2D binary random map (1: occupied, 0: free)
    - resolution: Resolution of the random map

    Returns:
    - esdf: Euclidean Signed Distance Field
    """
    return generate_esdf(random_map, resolution,enforce_obstable)

def visualize_esdf(esdf, resolution):
    """
    Visualize Euclidean Signed Distance Field (ESDF).

    Parameters:
    - esdf: Euclidean Signed Distance Field
    - resolution: Resolution of the ESDF
    """
    #展示原地图
    plt.imshow(random_map, extent=(0, random_map.shape[1] * resolution, 0, random_map.shape[0] * resolution))
    plt.colorbar(label='Signed Distance (meters)')
    plt.title('Random Map')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.show()
    #展示膨胀后的地图
    plt.imshow(esdf, extent=(0, esdf.shape[1] * resolution, 0, esdf.shape[0] * resolution))
    plt.colorbar(label='Signed Distance (meters)')
    plt.title('Euclidean Signed Distance Field (ESDF)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.show()

# Example usage:
size = 64
num_obstacles = 10
resolution = 1.0

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
csv_file_path = os.path.join(parent_directory, 'esdf_0.csv')#100指的是对障碍物点特殊处理的部分
# csv_file_path="test.csv"#测试下生成的速度
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    #生成一些数据，并保存在csv文件
    for i in range(1000):
        print(f"Generating map {i}...")
        random_map, flata_map = generate_random_map(size, num_obstacles)
        start_time=time.perf_counter()
        esdf = generate_esdf_from_random_map(flata_map, resolution,False)#True指的是对障碍物点特殊处理的部分
        # visualize_esdf(esdf, resolution)

        end_time=time.perf_counter()
        print(f"Time elapsed: {end_time - start_time:.15f} seconds")
        csv_writer.writerow([f"Map_{i}"])
        for i in range(random_map.shape[0]):
            csv_writer.writerow(random_map[i])
        #先写入一行名称
        csv_writer.writerow([f"Esdf_{i}"])
        for i in range(esdf.shape[0]):
            csv_writer.writerow(esdf[i])