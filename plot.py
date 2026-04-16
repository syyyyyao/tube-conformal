import matplotlib.pyplot as plt
import trimesh
import numpy as np
from src import raw_extension, ring_smooth

# data_path = 'data/real/H_AO_MFS_0027.obj' # -10, -110, -55
# data_path = 'data/real/H_AO_H_0159.obj' # 0, 120, 20
# data_path = 'data/real/H_ABAO_H_0190.obj' # 30, -80, -80
# data_path = 'data/real/H_ABAO_AAA_0193.obj' # 30, -60, -80

# data_path = 'data/synthetic/bent_02.obj' # 25, -85, -10
data_path = 'data/synthetic/straight_noise_01.obj' # 30, -90, -90
# data_path = 'data/synthetic/tapered_noise_01.obj' # 30, -90, -90
# data_path = 'data/synthetic/wavy_01.obj' # 30, -90, -90

mesh = trimesh.load(data_path)
v = np.asarray(mesh.vertices)
f = np.asarray(mesh.faces)
v_raw, f = raw_extension(v, f, 0.15)
v = ring_smooth(v_raw, f, 0.5)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, color="#d9d9d9", edgecolor="k", linewidth=0.1)
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=-10, azim=-110,roll=-55)
ax.set_xlim([v[:,0].min(), v[:,0].max()])
ax.set_ylim([v[:,1].min(), v[:,1].max()]) 
ax.set_zlim([v[:,2].min(), v[:,2].max()]) 
ax.set_aspect('equal')
ax.axis('off')
# plt.savefig("H_AO_MFS_0027.svg")
plt.show()