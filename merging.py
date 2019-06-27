import numpy as np
from skimage.future import graph


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])


def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst) / count
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass


def merge_regions(image, labels, edges=None, thresh=8.0, weight_mode='distance', merge_mode='merge_hierarchical'):
    if edges:
        g = graph.rag_boundary(labels, edges)
        labels2 = graph.merge_hierarchical(labels, g, thresh=thresh, rag_copy=False,
                                           in_place_merge=True,
                                           merge_func=merge_boundary,
                                           weight_func=weight_boundary)
    else:
        g = graph.rag_mean_color(image, labels, mode=weight_mode)
        if merge_mode == 'merge_hierarchical':
            labels2 = graph.merge_hierarchical(labels, g, thresh=thresh, rag_copy=False,
                                               in_place_merge=True,
                                               merge_func=merge_mean_color,
                                               weight_func=_weight_mean_color)
        elif merge_mode == 'cut_normalized':
            labels2 = graph.cut_normalized(labels, g)

        elif merge_mode == 'cut_threshold':
            labels2 = graph.cut_threshold(labels, g, thresh)
        else:
            raise ValueError("The merge mode '%s' is not recognised" % merge_mode)

    return labels2
