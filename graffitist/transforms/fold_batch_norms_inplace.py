#! /usr/bin/env python 3.6 (3379)
#coding=utf-8
# Compiled at: 2019-10-18 19:07:52
#Powered by BugScaner
#http://tools.bugscaner.com/
#如果觉得不错,请分享给你朋友使用吧!
"""
Removes batch normalization ops by folding them into convolutions.

Pre-requisite: Run 'freeze_graph' first!

@ author: Sambhav Jain
"""
__all__ = [
 'fold_batch_norms_inplace']
import re, math, numpy as np, tensorflow as tf
from graffitist.utils import graph_utils
INPUT_ORDER = {'BatchNormWithGlobalNormalization':[
  'conv_op', 'mean_op', 'var_op', 'beta_op', 'gamma_op'], 
 'FusedBatchNorm':[
  'conv_op', 'gamma_op', 'beta_op', 'mean_op', 'var_op']}
EPSILON_ATTR = {'BatchNormWithGlobalNormalization':'variance_epsilon', 
 'FusedBatchNorm':'epsilon'}

def scale_after_normalization(node):
    """
    Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/tools/optimize_for_inference_lib.py#L197-L201
    """
    if node.op == 'BatchNormWithGlobalNormalization':
        return node.attr['scale_after_normalization'].b
    else:
        return True


def fold_batch_norms_inplace(input_graph_def):
    """Removes batch normalization ops by folding them into convolutions.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/tools/optimize_for_inference_lib.py#L204-L368
    
    Batch normalization during training has multiple dynamic parameters that are
    updated, but once the graph is finalized these become constants. That means
    there's an opportunity to reduce the computations down to a scale and
    addition, rather than the more expensive multiple ops, and even bake the
    scaling into the convolution weights. This function identifies the typical
    pattern of batch normalization subgraphs, and performs the transformation to
    fold the computations down into a simpler form. It currently only spots batch
    normalization that's performed by the BatchNormWithGlobalNormalization op, and
    will need to be extended in the future to handle the newer style.
    
    Args:
      input_graph_def: A GraphDef containing a model.
    
    Returns:
      Modified graph with BN ops removed, and modified weights.
    
    Raises:
      ValueError: If the graph is badly formed with duplicate node names.
    """
    input_node_map = graph_utils.create_node_map(input_graph_def)
    nodes_to_skip = {}
    new_ops = []
    for node in input_graph_def.node:
        if node.op not in ('BatchNormWithGlobalNormalization', 'FusedBatchNorm'):
            continue
        conv_op = graph_utils.node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('conv_op')])
        if conv_op.op != 'Conv2D':
            print("WARNING: Didn't find expected Conv2D input to '%s'" % node.name)
            continue
            weights_op = graph_utils.node_from_map(input_node_map, conv_op.input[1])
            if weights_op.op != 'Const':
                print("WARNING: Didn't find expected conv Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (
                 conv_op.name, weights_op))
                continue
                weights = graph_utils.values_from_const(weights_op)
                channel_count = weights.shape[3]
                mean_op = graph_utils.node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('mean_op')])
                if mean_op.op != 'Const':
                    print("WARNING: Didn't find expected mean Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (
                     node.name, mean_op))
                    continue
                    mean_value = graph_utils.values_from_const(mean_op)
                    if mean_value.shape != (channel_count,):
                        print('WARNING: Incorrect shape for mean, found %s, expected %s, for node %s' % (
                         str(mean_value.shape),
                         str((
                          channel_count,)), node.name))
                        continue
                        var_op = graph_utils.node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('var_op')])
                        if var_op.op != 'Const':
                            print("WARNING: Didn't find expected var Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (
                             node.name, var_op))
                            continue
                            var_value = graph_utils.values_from_const(var_op)
                            if var_value.shape != (channel_count,):
                                print('WARNING: Incorrect shape for var, found %s, expected %s, for node %s' % (
                                 str(var_value.shape),
                                 str((
                                  channel_count,)), node.name))
                                continue
                                beta_op = graph_utils.node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('beta_op')])
                                if beta_op.op != 'Const':
                                    print("WARNING: Didn't find expected beta Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (
                                     node.name, beta_op))
                                    continue
                                    beta_value = graph_utils.values_from_const(beta_op)
                                    if beta_value.shape != (channel_count,):
                                        print('WARNING: Incorrect shape for beta, found %s, expected %s, for node %s' % (
                                         str(beta_value.shape),
                                         str((
                                          channel_count,)), node.name))
                                        continue
                                        gamma_op = graph_utils.node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('gamma_op')])
                                        if gamma_op.op != 'Const':
                                            print("WARNING: Didn't find expected gamma Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (
                                             node.name, gamma_op))
                                            continue
                                            gamma_value = graph_utils.values_from_const(gamma_op)
                                            if gamma_value.shape != (channel_count,):
                                                print('WARNING: Incorrect shape for gamma, found %s, expected %s, for node %s' % (
                                                 str(gamma_value.shape),
                                                 str((
                                                  channel_count,)), node.name))
                                                continue
                                                variance_epsilon_value = node.attr[EPSILON_ATTR[node.op]].f
                                                nodes_to_skip[node.name] = True
                                                nodes_to_skip[weights_op.name] = True
                                                nodes_to_skip[mean_op.name] = True
                                                nodes_to_skip[var_op.name] = True
                                                nodes_to_skip[beta_op.name] = True
                                                nodes_to_skip[gamma_op.name] = True
                                                nodes_to_skip[conv_op.name] = True
                                                if scale_after_normalization(node):
                                                    scale_value = 1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value) * gamma_value
                                                else:
                                                    scale_value = 1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value)
                                                offset_value = -mean_value * scale_value + beta_value
                                                scaled_weights = np.copy(weights)
                                                it = np.nditer(scaled_weights,
                                                  flags=['multi_index'], op_flags=['readwrite'])
                                                while not it.finished:
                                                    current_scale = scale_value[it.multi_index[3]]
                                                    it[0] *= current_scale
                                                    it.iternext()

                                                scaled_weights_op = tf.NodeDef()
                                                scaled_weights_op.op = 'Const'
                                                scaled_weights_op.name = weights_op.name
                                                scaled_weights_op.attr['dtype'].CopyFrom(weights_op.attr['dtype'])
                                                scaled_weights_op.attr['value'].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(scaled_weights, weights.dtype.type, weights.shape)))
                                                new_conv_op = tf.NodeDef()
                                                new_conv_op.CopyFrom(conv_op)
                                                offset_op = tf.NodeDef()
                                                offset_op.op = 'Const'
                                                offset_op.name = conv_op.name + '_bn_offset'
                                                offset_op.attr['dtype'].CopyFrom(mean_op.attr['dtype'])
                                                offset_op.attr['value'].CopyFrom(tf.AttrValue(tensor=tf.make_tensor_proto(offset_value, mean_value.dtype.type, offset_value.shape)))
                                                bias_add_op = tf.NodeDef()
                                                bias_add_op.op = 'BiasAdd'
                                                bias_add_op.name = node.name
                                                bias_add_op.attr['T'].CopyFrom(conv_op.attr['T'])
                                                bias_add_op.attr['data_format'].CopyFrom(conv_op.attr['data_format'])
                                                bias_add_op.input.extend([new_conv_op.name, offset_op.name])
                                                new_ops.extend([scaled_weights_op, new_conv_op, offset_op, bias_add_op])

    result_graph_def = tf.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.NodeDef()
        new_node.CopyFrom(node)
        result_graph_def.node.extend([new_node])

    result_graph_def.node.extend(new_ops)
    return result_graph_def