import numpy as np

def signed_dist(x, theta, theta_0):
    
    x= np.array(x).reshape(-1, 1)
    theta= np.array(theta).reshape(-1, 1)

    dot_product = np.dot(theta.T, x) + theta_0
    
    norm_theta = np.linalg.norm(theta)
    

    distance = (dot_product)/ norm_theta
    
    return distance[0]


x = np.array([[4], [-0.5]])
theta = np.array([[3], [4]])
theta_0 = 5


signed_distance = signed_dist(x, theta, theta_0)
print(signed_distance)
