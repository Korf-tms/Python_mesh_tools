# Post-processing tools for triangular 2d mesh
Works with node coordinates and element-node indices (can be generated e.g. using GMSH).

Jupyter notebooks with samples of meshes created using GMSH:
- gmsh_slope2d_coarse: simple 2d slope, coarse mesh
- gmsh_slope2d_fine: the same slope, fine mesh
- gmsh_slope2d_subdomains: the same slope, divided into subdomains
- gmsh_slope3d: slope without subdomains, extruded into 3d
- gmsh_slope3d_subdomains: slope with subdomains, extruded into 3d
- gmsh_slope3d_paper1: slope with thin layer according to [Seyed-Kolbadi, Sadoghi-Yazdi, Hariri-Ardebili, 2019]
- gmsh_slope3d_paper2: convex/concave slope according to [Yue Zhou, Shun-Chao Qi, Gang Fan, Ming-Liang Chen, Jia-Wen Zhou, 2020]

Other files:
- mesh_tools.py: post-processing tools for triangular 2d mesh
- mesh_tools_test.ipynb: Jupyter notebook testing the post-processing tools