import numpy as np


def random_2d(rng, min_radius=1, max_radius=1):
    # Sample a random angle uniformly from [0, 2*pi)
    angle = rng.uniform(0, 2 * np.pi)
    # Sample a random magnitude uniformly from [min_radius, max_radius]
    magnitude = rng.uniform(min_radius, max_radius)
    # Calculate the x and y components based on the angle and magnitude
    x = magnitude * np.cos(angle)
    y = magnitude * np.sin(angle)
    return x, y


def sort_vectors(vectors):
    # Sort the vectors based on the y-coordinate (descending) and then x-coordinate (ascending)
    sorted_vectors = sorted(vectors, key=lambda vec: (-vec[1], vec[0]))
    return sorted_vectors


def uniform_2d_numpy(interval, angles_range=(0, 360)):
    # Sample angles uniformly between 0 and 360 degrees with the specified interval
    angles_deg = np.arange(angles_range[0], angles_range[1], interval)
    angles_rad = np.deg2rad(angles_deg)
    # Convert angles to actions: each action is (cos(angle), sin(angle))
    sampled_actions = np.stack((np.cos(angles_rad), np.sin(angles_rad)), axis=1)
    return sampled_actions.tolist()


def vec2deg(vector):
    x, y = vector[0], vector[1]
    angle_radians = np.arctan2(y, x)  # Get the angle in radians
    angle_degrees = np.degrees(angle_radians)  # Convert radians to degrees
    return angle_degrees


def deg2vec(angle_deg):
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle_deg)
    # Calculate the cosine and sine of the angle and return as a NumPy array
    vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    return vector


def angle(v1, v2):
    """
    Calculate the positive angle in degrees between two vectors.
    """
    v1, v2 = np.array(v1), np.array(v2)
    dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Calculate the angle in degrees
    return np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))


def rotate_vec(vector, degrees):
    angle_rad = np.radians(degrees)
    # Create the rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return np.dot(rotation_matrix, np.asarray(vector))
