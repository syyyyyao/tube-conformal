from src import initial_tube, tube_conformal_map, extend_mesh, conformal_bend_major, conformal_bend_minor
import trimesh
import numpy as np

import matplotlib.pyplot as plt

from angular_distortion import angular_distortion


# data_path = 'data/real/sophie.obj'
data_path = 'data/real/amoeba1.obj'
# data_path = 'data/real/lionvase.obj'
# data_path = 'data/real/niccolo.obj'

mesh = trimesh.load(data_path)
v = np.asarray(mesh.vertices)
f = np.asarray(mesh.faces)


tube0 = initial_tube(v,f)
tube_fixed = tube_conformal_map(tube0, f, v, seam_strip_width=0.08)

v_ext, f_ext = extend_mesh(v, f, 2) 
tube0_ext = initial_tube(v_ext,f_ext)
tube_ext = tube_conformal_map(tube0_ext, f_ext, v_ext, seam_strip_width=0.08)
tube_free = tube_ext[:len(v)]


R = 2

tube_bent = conformal_bend_major(tube_free, R)
# tube_bent = conformal_bend_minor(tube_free, R)




dis_init = angular_distortion(v,f,tube0)
dis_fixed = angular_distortion(v,f,tube_fixed)
dis_free = angular_distortion(v,f,tube_free)
dis_bent = angular_distortion(v,f,tube_bent)

print(np.mean(np.abs(dis_init)))
print(np.mean(np.abs(dis_fixed)))
print(np.mean(np.abs(dis_free)))
print(np.mean(np.abs(dis_bent)))


# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_trisurf(tube_bent[:, 0], tube_bent[:, 1], tube_bent[:, 2], triangles=f, color="cyan", edgecolor="k", linewidth=0.2)
# ax.set_title("Tube Conformal Map")
# ax.set_box_aspect((1, 1, 1))
# ax.set_xlim([tube_bent[:,0].min(), tube_bent[:,0].max()])
# ax.set_ylim([tube_bent[:,1].min(), tube_bent[:,1].max()]) 
# ax.set_zlim([tube_bent[:,2].min(), tube_bent[:,2].max()]) 
# ax.set_aspect('equal')
# # ax.axis('off')
# plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(tube_fixed[:, 0], tube_fixed[:, 1], tube_fixed[:, 2], triangles=f, color="cyan", edgecolor="k", linewidth=0.2)
ax.set_title("Tube Conformal Map")
ax.set_box_aspect((1, 1, 1))
ax.set_xlim([tube_fixed[:,0].min(), tube_fixed[:,0].max()])
ax.set_ylim([tube_fixed[:,1].min(), tube_fixed[:,1].max()]) 
ax.set_zlim([tube_fixed[:,2].min(), tube_fixed[:,2].max()]) 
ax.set_aspect('equal')
# ax.axis('off')
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(tube0[:, 0], tube0[:, 1], tube0[:, 2], triangles=f, color="cyan", edgecolor="k", linewidth=0.2)
ax.set_title("Tube Conformal Map")
ax.set_box_aspect((1, 1, 1))
ax.set_xlim([tube0[:,0].min(), tube0[:,0].max()])
ax.set_ylim([tube0[:,1].min(), tube0[:,1].max()]) 
ax.set_zlim([tube0[:,2].min(), tube0[:,2].max()]) 
ax.set_aspect('equal')
# ax.axis('off')
plt.show()


