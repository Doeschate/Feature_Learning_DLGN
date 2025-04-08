import numpy as np

def set_seed(seed):
    np.random.seed(seed)

def generate_odt_data(num_data=1000, dim=2, seed=0, num_levels=2, threshold=0.0, w_list=None, b_list=None, vals=None):
    """
    Generate synthetic data labeled by an Oblique Decision Tree (ODT).

    Parameters:
    - num_data: Number of data points
    - dim: Input feature dimension
    - seed: Random seed
    - num_levels: Depth of the ODT (tree has 2^num_levels leaves)
    - threshold: Margin threshold for pruning points near decision boundaries
    - w_list, b_list: Predefined hyperplanes for internal nodes
    - vals: Labels for leaf nodes (default: alternating 0,1)

    Returns:
    - data_x: Input features
    - labels: Labels assigned by traversing the tree
    - tree_params: (w_list, b_list, vals)
    - stats: Number of points per node
    """
    set_seed(seed)
    num_internal_nodes = 2**num_levels - 1
    num_leaf_nodes = 2**num_levels
    stats = np.zeros(num_internal_nodes + num_leaf_nodes)

    if vals is None:
        vals = np.arange(0,num_internal_nodes+num_leaf_nodes,1,dtype=np.int32) % 2
        vals[:num_internal_nodes] = -99

    if w_list is None:
        w_list = np.random.standard_normal((num_internal_nodes, dim))
        w_list /= np.linalg.norm(w_list, axis=1)[:, None]
        b_list = np.zeros(num_internal_nodes)

    data_x = np.random.standard_normal((num_data, dim))
    data_x /= np.sqrt(np.sum(data_x**2, axis=1, keepdims=True))

    relevant_stats = data_x @ w_list.T + b_list
    curr_index = np.zeros(shape=(num_data), dtype=int)

    for level in range(num_levels):
        for el in range(2**level - 1, 2**(level + 1) - 1):
            relevant_stats[:, el] += b_list[el]
        decision = np.choose(curr_index, relevant_stats.T)
        curr_index = (curr_index + 1) * 2 - (1 - (decision > 0))

    bound_dist = np.min(np.abs(relevant_stats), axis=1)
    data_x_pruned = data_x[bound_dist > threshold]
    labels_pruned = vals[curr_index][bound_dist > threshold]

    relevant_stats = np.sign(data_x_pruned @ w_list.T + b_list)
    nodes_active = np.zeros((len(data_x_pruned), num_internal_nodes + num_leaf_nodes), dtype=int)

    for node in range(num_internal_nodes + num_leaf_nodes):
        if node == 0:
            stats[node] = len(relevant_stats)
            nodes_active[:, 0] = 1
            continue
        parent = (node - 1) // 2
        nodes_active[:, node] = nodes_active[:, parent]
        right_child = node-(parent*2)-1
        if right_child==1:
            nodes_active[:,node] *= relevant_stats[:,parent]>0
        if right_child==0:
            nodes_active[:,node] *= relevant_stats[:,parent]<0
        stats = nodes_active.sum(axis=0)

    return ((data_x_pruned, labels_pruned), (w_list, b_list, vals), stats)

# --- Example Usages (Uncomment as needed) ---

# Dataset I
# ((data_x, labels), (w_list, b_list, vals), stats) = generate_odt_data(
#     num_data=40000, dim=20, num_levels=4, seed=365
# )
# print(data_x.shape, labels.shape)

# Dataset II
# ((data_x, labels), (w_list, b_list, vals), stats) = generate_odt_data(
#     num_data=60000, dim=100, num_levels=4, seed=365
# )

# Dataset III
# ((data_x, labels), (w_list, b_list, vals), stats) = generate_odt_data(
#     num_data=100000, dim=500, num_levels=4, seed=365
# )
