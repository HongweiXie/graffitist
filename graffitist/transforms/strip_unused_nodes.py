#! /usr/bin/env python 3.6 (3379)
#coding=utf-8
# Compiled at: 2019-10-18 19:07:52
#Powered by BugScaner
#http://tools.bugscaner.com/
#如果觉得不错,请分享给你朋友使用吧!
"""
Removes unneeded nodes from a GraphDef file.

@ author: Sambhav Jain
"""
__all__ = [
 'strip_unused_nodes']
import copy, six, tensorflow as tf

def _node_name(n):
    if n.startswith('^'):
        return n[1:]
    else:
        return n.split(':')[0]


def _extract_graph_summary(graph_def):
    """Extracts useful information from the graph and returns them.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/0f486fc67070ba888204741c404a55a5f1a41fbc/tensorflow/python/framework/graph_util_impl.py#L116-L131
    """
    name_to_input_name = {}
    name_to_node = {}
    name_to_seq_num = {}
    seq = 0
    for node in graph_def.node:
        n = _node_name(node.name)
        name_to_node[n] = node
        name_to_input_name[n] = [_node_name(x) for x in node.input]
        name_to_seq_num[n] = seq
        seq += 1

    return (name_to_input_name, name_to_node, name_to_seq_num)


def _assert_nodes_are_present(name_to_node, nodes):
    """Assert that nodes are present in the graph.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/0f486fc67070ba888204741c404a55a5f1a41fbc/tensorflow/python/framework/graph_util_impl.py#L134-L137
    """
    for d in nodes:
        if not d in name_to_node:
            raise AssertionError('%s is not in graph' % d)


def _bfs_for_reachable_nodes(target_nodes, name_to_input_name):
    """Breadth first search for reachable nodes from target nodes.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/0f486fc67070ba888204741c404a55a5f1a41fbc/tensorflow/python/framework/graph_util_impl.py#L140-L154
    """
    nodes_to_keep = set()
    next_to_visit = target_nodes[:]
    while next_to_visit:
        node = next_to_visit[0]
        del next_to_visit[0]
        if node in nodes_to_keep:
            continue
        nodes_to_keep.add(node)
        if node in name_to_input_name:
            next_to_visit += name_to_input_name[node]

    return nodes_to_keep


def _extract_sub_graph(graph_def, dest_nodes):
    """Extract the subgraph that can reach any of the nodes in 'dest_nodes'.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/0f486fc67070ba888204741c404a55a5f1a41fbc/tensorflow/python/framework/graph_util_impl.py#L161-L195
    
    Args:
      graph_def: A graph_pb2.GraphDef proto.
      dest_nodes: A list of strings specifying the destination node names.
    Returns:
      The GraphDef of the sub-graph.
    
    Raises:
      TypeError: If 'graph_def' is not a graph_pb2.GraphDef proto.
    """
    if not isinstance(graph_def, tf.compat.v1.GraphDef):
        raise TypeError('graph_def must be a graph_pb2.GraphDef proto.')
    if isinstance(dest_nodes, six.string_types):
        raise TypeError('dest_nodes must be a list.')
    name_to_input_name, name_to_node, name_to_seq_num = _extract_graph_summary(graph_def)
    _assert_nodes_are_present(name_to_node, dest_nodes)
    nodes_to_keep = _bfs_for_reachable_nodes(dest_nodes, name_to_input_name)
    nodes_to_keep_list = sorted(list(nodes_to_keep),
      key=lambda n: name_to_seq_num[n])
    out = tf.compat.v1.GraphDef()
    for n in nodes_to_keep_list:
        out.node.extend([copy.deepcopy(name_to_node[n])])

    out.library.CopyFrom(graph_def.library)
    out.versions.CopyFrom(graph_def.versions)
    return out


def strip_unused_nodes(input_graph_def, input_node_names, output_node_names, placeholder_type_enum=[
 tf.float32.as_datatype_enum]):
    """Removes unused nodes from a GraphDef.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/tools/strip_unused_lib.py#L32-L87
    
    This script is designed to help streamline models, by taking the input and
    output nodes that will be used by an application and figuring out the smallest
    set of operations that are required to run for those arguments. The resulting
    minimal graph is then saved out.
    
    The advantages of running this script are:
    - You may be able to shrink the file size.
    - Operations that are unsupported on your platform but still present can be
     safely removed.
    
    The resulting graph may not be as flexible as the original though, since any
    input nodes that weren't explicitly mentioned may not be accessible any more.
    
    Args:
      input_graph_def: A graph with nodes we want to prune.
      input_node_names: A list of the nodes we use as inputs.
      output_node_names: A list of the output nodes.
      placeholder_type_enum: The AttrValue enum for the placeholder data type, or
        a list that specifies one value per input node name.
    
    Returns:
      A `GraphDef` with all unnecessary ops removed.
    
    Raises:
      ValueError: If any element in `input_node_names` refers to a tensor instead
       of an operation.
      KeyError: If any element in `input_node_names` is not found in the graph.
    """
    for name in input_node_names:
        if ':' in name:
            raise ValueError("Name '%s' appears to refer to a Tensor, not a Operation." % name)

    if isinstance(placeholder_type_enum, str):
        placeholder_type_enum = eval(placeholder_type_enum)
    not_found = {name for name in input_node_names}
    inputs_replaced_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in input_node_names:
            not_found.remove(node.name)
            placeholder_node = tf.compat.v1.NodeDef()
            placeholder_node.op = 'Placeholder'
            placeholder_node.name = node.name
            if isinstance(placeholder_type_enum, list):
                input_node_index = input_node_names.index(node.name)
                placeholder_node.attr['dtype'].CopyFrom(tf.compat.v1.AttrValue(type=placeholder_type_enum[input_node_index]))
            else:
                placeholder_node.attr['dtype'].CopyFrom(tf.compat.v1.AttrValue(type=placeholder_type_enum))
            if '_output_shapes' in node.attr:
                placeholder_node.attr['_output_shapes'].CopyFrom(node.attr['_output_shapes'])
            inputs_replaced_graph_def.node.extend([placeholder_node])
        else:
            inputs_replaced_graph_def.node.extend([copy.deepcopy(node)])

    if not_found:
        raise KeyError('The following input nodes were not found: %s\n' % not_found)
    output_graph_def = _extract_sub_graph(inputs_replaced_graph_def, output_node_names)
    return output_graph_def