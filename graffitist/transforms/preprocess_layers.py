#! /usr/bin/env python 3.6 (3379)
#coding=utf-8
# Compiled at: 2019-10-23 20:32:36
#Powered by BugScaner
#http://tools.bugscaner.com/
#如果觉得不错,请分享给你朋友使用吧!
"""
Preprocess layers based on PL implementation (e.g. avgpool, matmul, eltwise).

@ author: Sambhav Jain
"""
__all__ = [
 'preprocess_layers']
import re, numpy as np, tensorflow as tf
from graffitist.utils import graph_utils

def preprocess_layers(input_graph_def):
    output_graph_def = input_graph_def
    output_graph_def = remove_identity_nodes(output_graph_def)
    output_graph_def = preprocess_reduce_mean_nodes(output_graph_def)
    output_graph_def = preprocess_avgpool_nodes(output_graph_def)
    output_graph_def = preprocess_leakyrelu_nodes(output_graph_def)
    output_graph_def = collapse_concat_of_concat_nodes(output_graph_def)
    output_graph_def = rescale_eltwise_and_concat_nodes(output_graph_def)
    return output_graph_def


def remove_identity_nodes(input_graph_def):
    """
    Splices out identity nodes (if not involved in control edges, and not
    part of tf variables ('read'))
    
    This is also done by 'remove_training_nodes', however when qil retraining
    we don't use 'remove_training_nodes' as it also removes useful training nodes.
    
    In such cases 'remove_identity_nodes' helps remove identity nodes.
    
    An example where this is helpful is when generating qil training graph for mobilenet-v2,
    which contains lots of identity nodes that break the pattern matching of many layers
    such as eltwise_rescale etc.
    
    Always run this prior to 'rescale_eltwise_and_concat_nodes', which does add some 
    useful identity nodes (to rescale).
    """
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for node in input_graph_def.node:
        if node.op == 'Identity':
            if node.name.rpartition('/')[-1] != 'read':
                has_control_edge = False
                for input_name in node.input:
                    if re.match('^\\^', input_name):
                        has_control_edge = True

                if not has_control_edge:
                    nodes_to_skip[node.name] = True
                    consumer_nodes = output_node_map[node.name]
                    for consumer_node_name, input_index in consumer_nodes.items():
                        consumer_node = node_map[consumer_node_name]
                        del consumer_node.input[input_index]
                        consumer_node.input.insert(input_index, node.input[0])

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def preprocess_reduce_mean_nodes(input_graph_def):
    """
    Does the following modifications to reduce-mean node:
    
    Convert reduce_mean op to global avgpool with stride equal to input spatial dimensions.
    
    Always run this prior to 'preprocess_avgpool_nodes', which operates on all avgpool
    nodes in the graph.
    """
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for node in input_graph_def.node:
        if node.op == 'Mean':
            _reducemean_convert_to_avgpool(node, nodes_to_skip, new_nodes)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def preprocess_avgpool_nodes(input_graph_def):
    """
    Does the following modifications to avgpool node:
    
    1) Convert avgpool to depthwise_conv2d to model fixed point on both multiplicands.
       - i.e. reciprocal and input
    
    ############ Deprecated - cast to fp64 is now done as a post-process step. #################
    #2) Typecast to double (fp64) with 52 mantissa bits to model DSP58 / PL impl for avgpool.
    
    #  Pseudocode for DSP58 / PL impl for 7x7 avgpool kernel:
    
    #    for i from 1 to 49:
    #      acc += (1/49)  *  in_i
    #             ------     ----
    #            18 bits  +  8 bits   =   26 bits
    
    #  Due to the accumulation of 26 bit numbers, fp32 is insufficient (only 23 mantissa bits).
    #  Hence we use fp64 datatype instead with 52 mantissa bits.
    ############################################################################################
    """
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for node in input_graph_def.node:
        if node.op == 'AvgPool':
            _avgpool_convert_to_depthwise_conv2d(node, nodes_to_skip, new_nodes)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def preprocess_leakyrelu_nodes(input_graph_def):
    """
    Does the following modifications to leaky-relu node:
    
    Unfuse leakyrelu op to max(alpha*x, x) to enable quantization of intermediate tensors.
    """
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for node in input_graph_def.node:
        if node.op == 'LeakyRelu':
            _unfuse_leakyrelu_op(node, nodes_to_skip, new_nodes)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def rescale_eltwise_and_concat_nodes(input_graph_def):
    r"""
    Does the following modifications to rescale branches leading to element-wise add 
    (resnet-like identity shortcuts or bypass connections) and concat nodes:
    
    1) Inserts rescale (identity) nodes to specific branches leading to eltwise add node
       that require a rescale.
    
    2) Inserts rescale (identity) nodes to specific branches leading to concat node
       that require a rescale.
    
      Specifically, for resnet-v1, identity shortcut branches (to eltwise add) are rescaled
      (eltw_rescale_quant) to match scale of main branch. Projection shortcuts are not
      rescaled (since they will be handled by sharing thresholds of existing quant layers
      following BA nodes).
    
      Branches leading to concat that do not use quant layers (e.g. maxpool in inception v4)
      are also rescaled (concat_rescale_quant).
    
      Note that this implementation assumes eltwise add is always after BiasAdd
      and prior to ReLU (e.g. in Resnet v1 or Mobilenet v2). Hence output of rescale
      in the eltwise case is always signed.
    
      Another assumption is concat is always after ReLU. Hence output of rescale in the
      concat case is always unsigned.
    
                   [ C = Conv; BA = BiasAdd; R = ReLU; M = MaxPool]
             |           |            |             C   |
             |   |
             BA  |
             |   |
             R   |
             |  [M]
             C   |
             |   |
             BA  RESCALE
             |  /
             | /
             ++   (eltwise add)
             |
             R
    
                      M
        R   R    R    |
        |   |    |   RESCALE
         \   \   /   /
            concat
    
      (Here, R can be ReLU or LeakyReLU.)
    
      Eventually, all incoming branches will be quantized using a shared threshold / scale factor
      to ensure tensors with matching scales are being added / concatenated.
    """
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for node in input_graph_def.node:
        if node.op == 'Add' and 'BatchNorm_Fold' not in node.name and 'Initializer' not in node.name:
            shape_set = set()
            valid_idxs = []
            eltw_add_scope, _, _ = node.name.rpartition('/')
            for index, input_name in enumerate(node.input):
                input_node = graph_utils.node_from_map(node_map, input_name)
                if input_node.op != 'BiasAdd':
                    if input_node.op != 'Add':
                        valid_idxs.append(index)
                    if eltw_add_scope not in input_node.name:
                        valid_idxs.append(index)
                try:
                    input_shape = (
                     input_node.attr['_output_shapes'].list.shape[0].dim[0].size,
                     input_node.attr['_output_shapes'].list.shape[0].dim[1].size,
                     input_node.attr['_output_shapes'].list.shape[0].dim[2].size,
                     input_node.attr['_output_shapes'].list.shape[0].dim[3].size)
                except:
                    shape_set = set()
                    valid_idxs = []
                    break

                shape_set.add(input_shape)

        if len(shape_set) == 1:
            if len(valid_idxs) != 0:
                _insert_rescale_nodes(node, valid_idxs, 'eltw_rescale', nodes_to_skip, new_nodes)
            else:
                if node.op == 'ConcatV2':
                    valid_idxs = []
                    concat_scope, _, _ = node.name.rpartition('/')
                    for index, input_name in enumerate(node.input[:-1]):
                        input_node = graph_utils.node_from_map(node_map, input_name)
                        if input_node.op != 'Relu':
                            if input_node.op != 'Maximum':
                                valid_idxs.append(index)
                            if concat_scope not in input_node.name:
                                valid_idxs.append(index)

                    if len(valid_idxs) != 0:
                        _insert_rescale_nodes(node, valid_idxs, 'concat_rescale', nodes_to_skip, new_nodes)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def collapse_concat_of_concat_nodes(input_graph_def):
    r"""
    Does the following modifications to concat-of-concat connections (e.g. inception-v3)
    if axis being concatenated are the same.
    
    1) Removes first concat and rewires its inputs to second concat's inputs.
    
       |  \ /   \ /  |                     |  | |   | |  |
       |   C     C   |        --->         |  | |   | |  |
        \   \   /   /                       \  \ \ / /  /
              C                                   C
              |                                   |
    
      At the moment this can only handle concat depth of two. Can be extended in case deeper
      chains are observed.
    """
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for node in input_graph_def.node:
        if node.op == 'ConcatV2':
            mapping = []
            for index, input_name in enumerate(node.input):
                input_node = graph_utils.node_from_map(node_map, input_name)
                if input_node.op == 'ConcatV2':
                    first_concat_axis = graph_utils.values_from_const(graph_utils.node_from_map(node_map, input_node.input[-1]))
                    second_concat_axis = graph_utils.values_from_const(graph_utils.node_from_map(node_map, node.input[-1]))
                    if first_concat_axis == second_concat_axis:
                        pass
                    mapping += [(index, input_node.input[:-1])]
                    nodes_to_skip[input_node.name] = True
                    nodes_to_skip[input_node.input[-1]] = True

            if len(mapping) > 0:
                _collapse_concat_of_concat_nodes(node, mapping, nodes_to_skip, new_nodes)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def _reducemean_convert_to_avgpool(reducemean_node, nodes_to_skip, new_nodes):
    reducemean_input_node = graph_utils.node_from_map(node_map, reducemean_node.input[0])
    reduction_index_node = graph_utils.node_from_map(node_map, reducemean_node.input[1])
    reduction_indices = graph_utils.values_from_const(reduction_index_node)
    np.testing.assert_array_equal(reduction_indices, [1, 2], err_msg=("Expected reduction indices to be [1 2] of (N,H,W,C); got {}. Node: '{}'").format(reduction_indices, reducemean_node.name))
    _, H, W, _ = [x.size for x in reducemean_input_node.attr['_output_shapes'].list.shape[0].dim]
    nodes_to_skip[reducemean_node.name] = True
    nodes_to_skip[reduction_index_node.name] = True
    scope, sep, name = reducemean_node.name.rpartition('/')
    temp_graph = tf.Graph()
    with temp_graph.as_default():
        with temp_graph.name_scope(scope + sep + name + sep):
            input_tensor = tf.compat.v1.placeholder(tf.float32)
            output_tensor = tf.nn.avg_pool2d(input_tensor, ksize=[H, W], strides=1, padding='VALID')
    replace_map = {}
    replace_map[input_tensor.op.node_def.name] = reducemean_input_node.name
    temp_graph_def = temp_graph.as_graph_def(add_shapes=True)
    temp_node_map = graph_utils.create_node_map(temp_graph_def)
    temp_output_node_map = graph_utils.create_output_node_map(temp_graph_def)
    for node in temp_graph_def.node:
        if node.op == 'Placeholder':
            temp_output_nodes = temp_output_node_map[node.name]
            for temp_output_node_name, input_index in temp_output_nodes.items():
                temp_output_node = temp_node_map[temp_output_node_name]
                del temp_output_node.input[input_index]
                temp_output_node.input.insert(input_index, replace_map[node.name])

            continue
        new_nodes.extend([node])

    consumer_nodes = output_node_map[reducemean_node.name]
    for consumer_node_name, input_index in consumer_nodes.items():
        consumer_node = node_map[consumer_node_name]
        del consumer_node.input[input_index]
        consumer_node.input.insert(input_index, output_tensor.op.node_def.name)


def _avgpool_convert_to_depthwise_conv2d(avgpool_node, nodes_to_skip, new_nodes):
    nodes_to_skip[avgpool_node.name] = True
    scope, sep, name = avgpool_node.name.rpartition('/')
    avgpool_input_node = graph_utils.node_from_map(node_map, avgpool_node.input[0])
    input_channels = int(avgpool_input_node.attr['_output_shapes'].list.shape[0].dim[3].size)
    kernel_size = list(avgpool_node.attr['ksize'].list.i)
    strides = list(avgpool_node.attr['strides'].list.i)
    padding = avgpool_node.attr['padding'].s
    channel_multiplier = 1
    temp_graph = tf.Graph()
    with temp_graph.as_default():
        with temp_graph.name_scope(scope + sep + name + sep):
            input_tensor = tf.compat.v1.placeholder(tf.float32)
            filter_tensor = tf.constant(1.0 / (kernel_size[1] * kernel_size[2]), dtype=tf.float32, shape=[
             kernel_size[1], kernel_size[2], input_channels, channel_multiplier],
              name='avgpool_reciprocal')
            output_tensor = tf.compat.v1.nn.depthwise_conv2d_native(input_tensor, filter_tensor, strides,
              padding, name=name + '_from_avgpool')
    replace_map = {}
    replace_map[input_tensor.op.node_def.name] = graph_utils.node_name_from_input(avgpool_node.input[0])
    temp_graph_def = temp_graph.as_graph_def(add_shapes=True)
    temp_node_map = graph_utils.create_node_map(temp_graph_def)
    temp_output_node_map = graph_utils.create_output_node_map(temp_graph_def)
    for node in temp_graph_def.node:
        if node.op == 'Placeholder':
            temp_output_nodes = temp_output_node_map[node.name]
            for temp_output_node_name, input_index in temp_output_nodes.items():
                temp_output_node = temp_node_map[temp_output_node_name]
                del temp_output_node.input[input_index]
                temp_output_node.input.insert(input_index, replace_map[node.name])

            continue
        new_nodes.extend([node])

    consumer_nodes = output_node_map[avgpool_node.name]
    for consumer_node_name, input_index in consumer_nodes.items():
        consumer_node = node_map[consumer_node_name]
        del consumer_node.input[input_index]
        consumer_node.input.insert(input_index, output_tensor.op.node_def.name)


def _unfuse_leakyrelu_op(leakyrelu_node, nodes_to_skip, new_nodes):
    nodes_to_skip[leakyrelu_node.name] = True
    scope, sep, name = leakyrelu_node.name.rpartition('/')
    leakyrelu_input_node = graph_utils.node_from_map(node_map, leakyrelu_node.input[0])
    alpha_val = leakyrelu_node.attr['alpha'].f
    temp_graph = tf.Graph()
    with temp_graph.as_default():
        with temp_graph.name_scope(scope + sep + name + sep):
            input_tensor = tf.compat.v1.placeholder(tf.float32)
            alpha = tf.constant(alpha_val, name='alpha')
            alpha_x = tf.math.multiply(input_tensor, alpha, name='mul')
            output_tensor = tf.math.maximum(alpha_x, input_tensor, name='Maximum')
    replace_map = {}
    replace_map[input_tensor.op.node_def.name] = graph_utils.node_name_from_input(leakyrelu_node.input[0])
    temp_graph_def = temp_graph.as_graph_def(add_shapes=True)
    temp_node_map = graph_utils.create_node_map(temp_graph_def)
    temp_output_node_map = graph_utils.create_output_node_map(temp_graph_def)
    for node in temp_graph_def.node:
        if node.op == 'Placeholder':
            temp_output_nodes = temp_output_node_map[node.name]
            for temp_output_node_name, input_index in temp_output_nodes.items():
                temp_output_node = temp_node_map[temp_output_node_name]
                del temp_output_node.input[input_index]
                temp_output_node.input.insert(input_index, replace_map[node.name])

            continue
        new_nodes.extend([node])

    consumer_nodes = output_node_map[leakyrelu_node.name]
    for consumer_node_name, input_index in consumer_nodes.items():
        consumer_node = node_map[consumer_node_name]
        del consumer_node.input[input_index]
        consumer_node.input.insert(input_index, output_tensor.op.node_def.name)


def _insert_rescale_nodes(node, valid_idxs, rescale_name, nodes_to_skip, new_nodes):
    scope, sep, name = node.name.rpartition('/')
    for index, input_name in enumerate(node.input):
        if index in valid_idxs:
            identity_node = tf.compat.v1.NodeDef()
            identity_node.op = 'Identity'
            identity_node.name = scope + sep + name + '_' + rescale_name + '_' + str(index)
            identity_node.attr['T'].CopyFrom(node.attr['T'])
            identity_node.input.extend([input_name])
            del node.input[index]
            node.input.insert(index, identity_node.name)
            new_nodes.extend([identity_node])


def _collapse_concat_of_concat_nodes(concat_node, mapping, nodes_to_skip, new_nodes):
    idx_offset = 0
    for m in mapping:
        idx = m[0] + idx_offset
        inputs = m[1]
        del concat_node.input[idx]
        concat_node.input[idx:idx] = inputs
        idx_offset += len(inputs) - 1

    concat_node.attr['N'].i += idx_offset