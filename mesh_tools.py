import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, find, triu


def rectangle_triangulation(nx, ny):
    idx = np.reshape(np.arange(nx*ny), (ny, nx))
    idx1 = idx[:-1, :-1].flatten()
    idx2 = idx[:-1, 1:].flatten()
    idx3 = idx[1:, 1:].flatten()
    idx4 = idx[1:, :-1].flatten()
    elem1 = np.repeat(idx1, 2)
    elem2 = np.repeat(idx2, 2)
    elem2[1::2] = idx3
    elem3 = np.repeat(idx3, 2)
    elem3[1::2] = idx4
    elem = np.vstack((elem1, elem2, elem3))
    return elem.T


def combinations_with_repetitions(arr, k, prefix=[], start=0, result=[]):
    if k == 0:
        result.append(prefix.copy())
        return
    for i in range(start, len(arr)):
        combinations_with_repetitions(arr, k - 1, prefix + [arr[i]], i, result)
    return result


def weights_inside_for_PN(N=5):
    k: int = N-3
    comb = combinations_with_repetitions([1, 2, 3], k, start=0, result=[])
    comb = np.array(comb)
    comb_with_1 = np.sum(comb == 1, axis=1).reshape((1, -1))
    comb_with_2 = np.sum(comb == 2, axis=1).reshape((1, -1))
    comb_with_3 = np.sum(comb == 3, axis=1).reshape((1, -1))
    weights = np.concatenate((comb_with_1, comb_with_2, comb_with_3), axis=0)
    return (1+weights)/N


class Mesh2d:
    """Triangular 2d mesh with post-processing tools."""

    def __init__(self, node_X, node_Y, elem):
        self.node_X = node_X
        self.node_Y = node_Y
        self.elem = elem
        self.update()

    def update(self):
        self.n_node = self.node_X.shape[0]
        self.n_elem = self.elem.shape[0]

        # sparse adjacency matrix - each row is one element, each column is one node
        # if a node belongs to an element, the corresponding matrix entry is 1; otherwise, it is 0
        row = np.arange(self.n_elem).repeat(3)
        col = self.elem.flatten()
        data = np.ones((3*self.n_elem,), dtype=int)
        self.elem_node_adj = coo_matrix((data, (row, col)), shape=(self.n_elem, self.n_node))

        # for each node, calculate number of adjacent elements
        self.n_elements_per_node = np.sum(self.elem_node_adj, axis=0)

        # for each pair of elements, calculate number of common nodes
        # obviously, 3 on diagonal
        self.elem_elem_adj = self.elem_node_adj @ self.elem_node_adj.T

        # for each pair of nodes, calculate number of elements adjacent to both
        # diagional entries show, how many elements are adjacent to that node (n_elements_per_node)
        # if a non-diagonal entry equals 1, these two nodes form a boundary edge
        # if a non-diagonal entry equals 2, these two nodes form an interior edge
        self.node_node_adj = self.elem_node_adj.T @ self.elem_node_adj

        # similarly to elem, this array contains node indices for each edge
        edges_matrix = triu(self.node_node_adj, 1)
        node0, node1, value = find(edges_matrix)
        self.edge = np.concatenate(([node0], [node1])).T
        self.n_edge = self.edge.shape[0]

        # indicates if the edge is on the boundary
        self.edge_boundary_flag = value == 1

        # coordinates of centers of elements:
        self.elem_center_X = np.mean(self.node_X[self.elem], axis=1)
        self.elem_center_Y = np.mean(self.node_Y[self.elem], axis=1)

        # coordinates of midpoints of edges:
        self.edge_mid_X = np.mean(self.node_X[self.edge], axis=1)
        self.edge_mid_Y = np.mean(self.node_Y[self.edge], axis=1)

        # sparse adjacency matrix - each row is one edge, each column is one node
        # if a node belongs to an edge, the corresponding matrix entry is 1; otherwise, it is 0
        row = np.arange(self.n_edge).repeat(2)
        col = self.edge.flatten()
        data = np.ones((2*self.n_edge,), dtype=int)
        self.edge_node_adj = coo_matrix((data, (row, col)), shape=(self.n_edge, self.n_node))

        # sparse adjacency matrix - each row is one element, each column is one edge
        # if an edge belongs to an element, the corresponding matrix entry is 2;
        # if they share only 1 node, the entry is 1; otherwise, it is 0
        self.elem_edge_adj_weighted = self.elem_node_adj @ self.edge_node_adj.T

        # sparse adjacency matrix - each row is one element, each column is one edge
        # if an edge belongs to an element, the corresponding matrix entry is 1; otherwise, it is 0
        self.elem_edge_adj = self.elem_edge_adj_weighted.copy()
        self.elem_edge_adj[self.elem_edge_adj_weighted == 1] = 0
        self.elem_edge_adj[self.elem_edge_adj_weighted == 2] = 1

        # matrix of edge indices for each element:
        # this way, edge ordering does not correspond to node ordering, they are sorted
        # elem, edge, value = find(self.elem_edge_adj)

        # for each element, edge 0 goes form node 0 to 1, 1 (1->2), 2(2->)
        # sparse adjacency matrix - each row is first (second, third) edge of each element, each column is one node
        self.elem_node_adj_partial = []
        for i in range(3):
            row = np.arange(self.n_elem).repeat(2)
            elem_edge_i = np.delete(self.elem, i-1, axis=1)
            col = elem_edge_i.flatten()
            data = np.ones((2*self.n_elem,), dtype=int)
            self.elem_node_adj_partial.append(coo_matrix((data, (row, col)), shape=(self.n_elem, self.n_node)))

    def nodes_on_edges_for_PN(self, N=2):
        x = self.node_X[self.edge]
        y = self.node_Y[self.edge]
        # P2: [1/2, 1/2]
        # P3: [1/3, 2/3], [2/3, 1/3]
        # P4: [1/4, 3/4], [2/4, 2/4], [3/4, 1/4]
        # P5: [1/5, 4/5], [2/5, 3/5], [3/5, 2/5], [4/5, 1/5]
        weights0 = np.arange(1, N).reshape((1, -1))/N
        weights = np.concatenate((1-weights0, weights0))
        new_node_X = x @ weights
        new_node_Y = y @ weights
        return new_node_X, new_node_Y

    def nodes_inside_elements_for_PN(self, N=3):
        if N < 3:
            return np.empty((self.n_elem, 0)),  np.empty((self.n_elem, 0))
        # P3: [1/3, 1/3, 1/3]
        # P4: [1/4, 1/4, 2/4], [1/4, 2/4, 1,4]
        # P5: [1/5, 1/5, 3/5], [1/5, 2/5, 2/5], [1/5, 3/5, 1/5], [2/5, 1/5, 2/5], [2/5, 2/5, 1/5], [3/5, 1/5, 1/5]
        if N == 3:
            weights = np.ones((3, 1))/3
        else:
            weights = weights_inside_for_PN(N)
        x = self.node_X[self.elem]
        y = self.node_Y[self.elem]
        new_node_X = x @ weights
        new_node_Y = y @ weights
        return new_node_X, new_node_Y

    def P1_to_PN(self, N):
        # PN node indices for edges
        new_node_X, new_node_Y = self.nodes_on_edges_for_PN(N)
        n_node_per_edge = new_node_X.shape[1]
        n_new_node = n_node_per_edge * self.n_edge
        self.edge_PN = np.arange(self.n_node, self.n_node+n_new_node).reshape((self.n_edge, n_node_per_edge))
        n_node = self.n_node+n_new_node
        self.node_X = np.concatenate((self.node_X, new_node_X.flatten()))
        self.node_Y = np.concatenate((self.node_Y, new_node_Y.flatten()))

        self.elem_PN_edge = np.empty((self.n_elem, 0), dtype=int)
        for i in range(3):
            # indices of first (second, third) edges:
            tmp = self.elem_node_adj_partial[i] @ self.edge_node_adj.T
            tmp[tmp == 1] = 0
            _, edge, _ = find(tmp)
            # compare order nodes in edges:
            edge_start = self.edge[edge, 0]
            elem_node_i = self.elem[:, i]

            this_edge = np.zeros((self.n_elem, n_node_per_edge))
            correct_ordering = edge_start == elem_node_i
            this_edge[correct_ordering, :] = self.edge_PN[edge[correct_ordering]]
            this_edge[~correct_ordering, :] = np.fliplr(self.edge_PN[edge[~correct_ordering]])

            self.elem_PN_edge = np.concatenate((self.elem_PN_edge, this_edge), axis=1)
            # TODO: permutation of node indices may be required

        # PN node indices for elements - nodes inside
        new_node_X, new_node_Y = self.nodes_inside_elements_for_PN(N)
        n_node_per_elem = new_node_X.shape[1]
        n_new_node = n_node_per_elem * self.n_elem
        self.elem_PN_inside = np.arange(n_node, n_node+n_new_node).reshape((self.n_elem, n_node_per_elem))
        self.node_X = np.concatenate((self.node_X, new_node_X.flatten()))
        self.node_Y = np.concatenate((self.node_Y, new_node_Y.flatten()))
        self.elem_PN = np.concatenate((self.elem, self.elem_PN_edge, self.elem_PN_inside), axis=1)
        self.update()

    def plot_mesh(self):
        plt.triplot(self.node_X, self.node_Y, self.elem)  # plot triangle edges
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Triangular mesh")
        plt.axis('equal')  # ensure equal aspect ratio

    def plot_nodes(self, show_numbers=False, markersize=5):
        plt.plot(self.node_X, self.node_Y, 'o', color="cyan", markersize=markersize)  # plot nodes
        if show_numbers:
            i = 0
            for x, y in zip(self.node_X, self.node_Y):
                plt.text(x, y, str(i))  # , va="center", ha="center")
                i += 1

    def plot_elem_centers(self, show_numbers=False, markersize=3):
        plt.plot(self.elem_center_X, self.elem_center_Y, 'o', color='orange', markersize=markersize)
        if show_numbers:
            i = 0
            for x, y in zip(self.elem_center_X, self.elem_center_Y):
                plt.text(x, y, str(i), va="center", ha="center")
                i += 1

    def plot_edge_midpoints(self, show_numbers=False, markersize=3):
        plt.plot(self.edge_mid_X, self.edge_mid_Y, 'o',  color="green", markersize=markersize)
        if show_numbers:
            i = 0
            for x, y in zip(self.edge_mid_X, self.edge_mid_Y):
                plt.text(x, y, str(i))
                i += 1
