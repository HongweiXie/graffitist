#! /usr/bin/env python 3.6 (3379)
#coding=utf-8
# Compiled at: 2019-10-18 19:07:52
#Powered by BugScaner
#http://tools.bugscaner.com/
#如果觉得不错,请分享给你朋友使用吧!
"""
Fix input placeholder shape explicitly, to allow the static shapes
for forthcoming nodes to be populated correctly. The static shape is
inferred and stored under the NodeDef attribute node.attr["_output_shapes"].

@ author: Sambhav Jain
"""
__all__ = [
 'fix_input_shape']
import tensorflow as tf
from graffitist.utils import graph_utils

def fix_input_shape(input_graph_def, input_node_names, input_shape):
    node_map = graph_utils.create_node_map(input_graph_def)
    input_shape = [int(x) for x in input_shape.split(',')]
    for input_node_name in input_node_names:
        if ':' in input_node_name:
            raise ValueError("Name '%s' appears to refer to a Tensor, not a Operation." % input_node_name)
        input_node = graph_utils.node_from_map(node_map, input_node_name)
        temp_graph = tf.Graph()
        with temp_graph.as_default():
            input_tensor = tf.compat.v1.placeholder(tf.float32, shape=[None] + input_shape)
        input_node.attr['shape'].CopyFrom(input_tensor.op.node_def.attr['shape'])

    for node in input_graph_def.node:
        if '_output_shapes' in node.attr:
            if node.op != 'ResizeNearestNeighbor':
                del node.attr['_output_shapes']

    output_graph_def = graph_utils.add_static_shapes(input_graph_def)
    return output_graph_def