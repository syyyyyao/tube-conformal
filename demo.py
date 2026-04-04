from src import initial_tube, tube_conformal_map, extend_mesh, conformal_bend_major, conformal_bend_minor
import trimesh
import numpy as np

import matplotlib.pyplot as plt

from angular_distortion import angular_distortion


# data_path = 'data/sophie.obj'
data_path = 'data/amoeba1.obj'
# data_path = 'data/lionvase.obj'
# data_path = 'data/niccolo.obj'

mesh = trimesh.load(data_path)
v = np.asarray(mesh.vertices)
f = np.asarray(mesh.faces)

v_ext, f_ext = extend_mesh(v, f, 2) 
tube0_ext = initial_tube(v_ext,f_ext)
tube_ext = tube_conformal_map(tube0_ext, f_ext, v_ext)
tube_free = tube0_ext[:len(v)]


R = 2

tube_bent = conformal_bend_major(tube_free, R)
# tube_bent = conformal_bend_minor(tube_free, R)

# conformal bending tube
# u = np.arctan2(tube_free[:,1], tube_free[:,0])

# theta = 2 * np.arctan2(np.sqrt(R+1) * np.sin(0.5*u),
#                        np.sqrt(R-1) * np.cos(0.5*u))


# phi = tube_free[:,2] / np.sqrt(R**2 - 1)

# torus_par = np.column_stack([
#     (R + np.cos(theta)) * np.cos(phi),
#     np.sin(theta),
#     (R + np.cos(theta)) * np.sin(phi)
# ])


# conformal bending annulus
# z = tube_free[:,2]
# z = z- (np.max(z) + np.min(z)) / 2

# theta = 2 * np.arctan2(np.sqrt(R+1)* np.sin(np.sqrt(R**2-1)/2 * z),
#                        np.sqrt(R-1)* np.cos(np.sqrt(R**2-1)/2 * z))

# phi = np.arctan2(tube_free[:,1], tube_free[:,0])

# torus_par = np.column_stack([
#     (R + np.cos(theta)) * np.cos(phi),
#     np.sin(theta),
#     (R + np.cos(theta)) * np.sin(phi)
# ])



dis_tube = angular_distortion(v,f,tube_free)
dis_bent = angular_distortion(v,f,tube_bent)

print(np.mean(np.abs(dis_tube)),np.mean(np.abs(dis_bent)))

dis = angular_distortion(tube_free, f, tube_bent)
print(np.mean(np.abs(dis)))


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(tube_bent[:, 0], tube_bent[:, 1], tube_bent[:, 2], triangles=f, color="cyan", edgecolor="k", linewidth=0.2)
ax.set_title("Tube Conformal Map")
ax.set_box_aspect((1, 1, 1))
ax.set_xlim([tube_bent[:,0].min(), tube_bent[:,0].max()])
ax.set_ylim([tube_bent[:,1].min(), tube_bent[:,1].max()]) 
ax.set_zlim([tube_bent[:,2].min(), tube_bent[:,2].max()]) 
ax.set_aspect('equal')
# ax.axis('off')
plt.show()




