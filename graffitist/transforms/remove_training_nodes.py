#! /usr/bin/env python 3.6 (3379)
#coding=utf-8
# Compiled at: 2019-10-18 19:07:52
#Powered by BugScaner
#http://tools.bugscaner.com/
#如果觉得不错,请分享给你朋友使用吧!
"""
Prunes out nodes that aren't needed for inference.

@ author: Sambhav Jain
"""
__all__ = [
 'remove_training_nodes']
import re, tensorflow as tf

def _remove_training_nodes(input_graph, protected_nodes=None):
    """Prunes out nodes that aren't needed for inference.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/0f486fc67070ba888204741c404a55a5f1a41fbc/tensorflow/python/framework/graph_util_impl.py#L372-L457
    
    There are nodes like Identity and CheckNumerics that are only useful
    during training, and can be removed in graphs that will be used for
    nothing but inference. Here we identify and remove them, returning an
    equivalent graph. To be specific, CheckNumerics nodes are always removed, and
    Identity nodes that aren't involved in control edges are spliced out so that
    their input and outputs are directly connected.
    
    Args:
      input_graph: Model to analyze and prune.
      protected_nodes: An optional list of names of nodes to be kept
        unconditionally. This is for example useful to preserve Identity output
        nodes.
    
    Returns:
      A list of nodes with the unnecessary ones removed.
    """
    if not protected_nodes:
        protected_nodes = []
    types_to_remove = {'CheckNumerics': True}
    input_nodes = input_graph.node
    names_to_remove = {}
    for node in input_nodes:
        if node.op in types_to_remove:
            names_to_remove[node.name] = node.name not in protected_nodes and True

    nodes_after_removal = []
    for node in input_nodes:
        if node.name in names_to_remove:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub('^\\^', '', full_input_name)
            if input_name in names_to_remove:
                continue
            new_node.input.append(full_input_name)

        nodes_after_removal.append(new_node)

    types_to_splice = {'Identity': True}
    control_input_names = set()
    node_names_with_control_input = set()
    for node in nodes_after_removal:
        for node_input in node.input:
            if '^' in node_input:
                control_input_names.add(node_input.replace('^', ''))
                node_names_with_control_input.add(node.name)

    names_to_splice = {}
    for node in nodes_after_removal:
        if node.op in types_to_splice:
            names_to_splice[node.name] = node.name not in protected_nodes and node.name not in node_names_with_control_input and node.input[0]

    names_to_splice = {name:value for name, value in names_to_splice.items() if name not in control_input_names}
    nodes_after_splicing = []
    for node in nodes_after_removal:
        if node.name in names_to_splice:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            input_name = re.sub('^\\^', '', full_input_name)
            while input_name in names_to_splice:
                full_input_name = names_to_splice[input_name]
                input_name = re.sub('^\\^', '', full_input_name)

            new_node.input.append(full_input_name)

        nodes_after_splicing.append(new_node)

    output_graph = tf.compat.v1.GraphDef()
    output_graph.node.extend(nodes_after_splicing)
    return output_graph


def remove_training_nodes(input_graph_def, protected_nodes, protected_node_patterns=[]):
    """
    There are nodes like Identity and CheckNumerics that are only useful
    during training, and can be removed in graphs that will be used for
    nothing but inference. Here we identify and remove them, returning an
    equivalent graph. To be specific, CheckNumerics nodes are always removed, and
    Identity nodes that aren't involved in control edges are spliced out so that
    their input and outputs are directly connected.
    
    Args:
      input_graph_def: Model to analyze and prune.
      protected_nodes: Optional list of names of nodes to be kept
        unconditionally. This is for example useful to preserve Identity output
        nodes.
      protected_node_patterns: Optional list of regex patterns matching node names
        that are to be protected / kept untouched.
    
    Returns:
      A modified graph_def with the unnecessary training nodes removed.
    """
    if isinstance(protected_node_patterns, str):
        protected_node_patterns = eval(protected_node_patterns)
    name_to_node = {}
    for node in input_graph_def.node:
        name_to_node[node.name] = node

    for n_name in protected_nodes:
        if not n_name in name_to_node:
            raise AssertionError('%s is not in graph' % n_name)

    for pattern in protected_node_patterns:
        for node in input_graph_def.node:
            if re.match(pattern, node.name):
                protected_nodes.append(node.name)

    output_graph_def = _remove_training_nodes(input_graph_def, protected_nodes)
    return output_graph_def