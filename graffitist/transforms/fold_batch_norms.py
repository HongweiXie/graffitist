#! /usr/bin/env python 3.6 (3379)
#coding=utf-8
# Compiled at: 2019-10-23 20:32:36
#Powered by BugScaner
#http://tools.bugscaner.com/
#如果觉得不错,请分享给你朋友使用吧!
"""
Fold FusedBatchNorm into preceding convolution or FC layer (without
modifying weights in place).

This is accomplished with additional nodes to do the folding. It also
works both with unfrozen and frozen graphs.

@ author: Sambhav Jain
"""
__all__ = [
 'fold_batch_norms']
import numpy as np, tensorflow as tf
from graffitist.utils import graph_utils, graph_matcher
from graffitist.transforms.preprocess_layers import remove_identity_nodes

def fold_batch_norms(input_graph_def, is_training=False):
    is_training = is_training == 'True' or is_training == True
    output_graph_def = remove_identity_nodes(input_graph_def)
    if is_training:
        output_graph_def = _fold_fused_batch_norms_training(output_graph_def)
    else:
        output_graph_def = _fold_fused_batch_norms_inference(output_graph_def)
    return output_graph_def


@tf.RegisterGradient('FoldFusedBatchNormGradient')
def _FoldFusedBatchNormGrad(op, unused_grad_y, grad_mean, grad_var, unused_1, unused_2):
    """The gradients for `FoldFusedBatchNorm`.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/quantize/python/fold_batch_norms.py#L445-L453
    
    Args:
      op: The `FoldFusedBatchNorm` `Operation` that we are differentiating, which we can use
          to find the inputs and outputs of the original op.
      unused_grad_y: Gradient with respect to the output[0] (y) of the `FoldFusedBatchNorm` op
                     (this is unused since FusedBatchNorm is folded and the main output y is unused,
                      hence no longer on the activation path).
      grad_mean: Gradient with respect to the output[1] (mean) of the `FoldFusedBatchNorm` op.
      grad_var: Gradient with respect to the output[2] (var) of the `FoldFusedBatchNorm` op.
    
    Returns:
      Gradients with respect to the inputs of `FoldFusedBatchNorm`.
    
    Once folded, the main output y is unused. Only the batch mean and batch variance outputs are
    used on the forward activation path. Hence in the backward pass (dL/dx), add the gradients along 
    these two paths to the input.
    
    i.e. 
        dL/dx = dL/dmean * dmean/dx + dL/dvar * dvar/dx
    
    where
        mean = sum(x, axis=-1)/n
        var = {(x-mean)^2 / n} * n/(n-1)
    
    Note that batch variance is Bessel corrected.
    
    Hence
        dL/dx = dL/dmean * dmean/dx + dL/dvar * dvar/dx
              = grad_mean * (1/n)   + grad_var * (2*(x-mean)/(n-1))
    """
    x = op.inputs[0]
    n = tf.cast(tf.size(x) / tf.size(grad_mean), tf.float32)
    dmean_dx = grad_mean / n
    dvar_dx = 2 * grad_var * (x - op.outputs[1]) / (n - 1)
    return (
     dmean_dx + dvar_dx, None, None, None, None)


def _fold_fused_batch_norms_training(input_graph_def):
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    temp_graph = tf.Graph()
    with temp_graph.as_default():
        freeze_bn_default = tf.constant(True, dtype=tf.bool, shape=[], name='freeze_bn_default')
        freeze_bn_bool_tensor = tf.compat.v1.placeholder_with_default(freeze_bn_default, shape=[], name='freeze_bn')
        freeze_bn_tensor = tf.cast(freeze_bn_bool_tensor, dtype=tf.float32, name='freeze_bn_cast')
    freeze_bn_default_node = freeze_bn_default.op.node_def
    freeze_bn_bool_node = freeze_bn_bool_tensor.op.node_def
    freeze_bn_node = freeze_bn_tensor.op.node_def
    new_nodes.extend([freeze_bn_default_node])
    new_nodes.extend([freeze_bn_bool_node])
    new_nodes.extend([freeze_bn_node])
    for match_result in graph_matcher.find_fused_batch_norms(input_graph_def):
        input_node = match_result.get_op('input_pattern')
        weight_node = match_result.get_op('weight_pattern')
        gamma_node = match_result.get_op('gamma_pattern')
        beta_node = match_result.get_op('beta_pattern')
        mean_mv_node = match_result.get_op('mean_pattern')
        variance_mv_node = match_result.get_op('variance_pattern')
        layer_node = match_result.get_op('layer_pattern')
        bn_node = match_result.get_op('batch_norm_pattern')
        final_output_nodes = output_node_map[bn_node.name]
        if bn_node.attr['is_training'].b is True:
            delete_keys = []
            for node_name, input_index in final_output_nodes.items():
                if 'AssignMovingAvg' in node_name:
                    moving_avg_sub_node = node_map[node_name]
                    moving_avg_node = graph_utils.node_from_map(node_map, moving_avg_sub_node.input[0])
                    if 'moving_mean' in moving_avg_node.name:
                        mean_mv_node = moving_avg_node
                        mean_b_node_name = moving_avg_sub_node.input[1]
                    else:
                        if 'moving_variance' in moving_avg_node.name:
                            variance_mv_node = moving_avg_node
                            variance_b_node_name = moving_avg_sub_node.input[1]
                    delete_keys.append(node_name)

            for key in delete_keys:
                del final_output_nodes[key]

        else:
            print('ERROR: FusedBatchNorm in inference mode cannot be folded for training graph; change to training mode and rerun.')
            return -1
        if layer_node.op == 'MatMul':
            matmul_reshape_node = match_result.get_op('matmul_reshape_pattern')
            matmul_bn_output_reshape_node = match_result.get_op('matmul_bn_output_reshape_pattern')
            final_output_nodes = output_node_map[matmul_bn_output_reshape_node.name]
        if layer_node.op == 'MatMul':
            nodes_to_skip[matmul_bn_output_reshape_node.name] = True
        if layer_node.op == 'DepthwiseConv2dNative':
            in_channels = weight_node.attr['_output_shapes'].list.shape[0].dim[2].size
            depth_multiplier = weight_node.attr['_output_shapes'].list.shape[0].dim[3].size
        bn_node.attr['_gradient_op_type'].CopyFrom(tf.compat.v1.AttrValue(s=tf.compat.as_bytes('FoldFusedBatchNormGradient')))
        cloned_layer_node = tf.compat.v1.NodeDef()
        cloned_layer_node.CopyFrom(layer_node)
        cloned_layer_node.name = cloned_layer_node.name + '_Fold'
        new_nodes.extend([cloned_layer_node])
        temp_graph = tf.Graph()
        with temp_graph.as_default():
            scope, sep, _ = bn_node.name.rpartition('/')
            with temp_graph.name_scope(scope + sep):
                layer_tensor = tf.compat.v1.placeholder(tf.float32)
                mean_b_tensor = tf.compat.v1.placeholder(tf.float32)
                var_b_tensor = tf.compat.v1.placeholder(tf.float32)
                n = tf.cast(tf.size(layer_tensor) / tf.size(mean_b_tensor), tf.float32)
                unbessel_var_b_tensor = tf.multiply(var_b_tensor, (n - 1) / n, name='Undo_Bessel_Correction')
            scope, sep, _ = layer_node.name.rpartition('/')
            with temp_graph.name_scope(scope + sep + 'BatchNorm_Fold' + sep):
                mean_mv_tensor = tf.compat.v1.placeholder(tf.float32)
                var_mv_tensor = tf.compat.v1.placeholder(tf.float32)
                beta_tensor = tf.compat.v1.placeholder(tf.float32)
                gamma_tensor = tf.compat.v1.placeholder(tf.float32)
                freeze_bn_tensor = tf.compat.v1.placeholder(tf.float32)
                eps_value = np.array(bn_node.attr['epsilon'].f, dtype=np.float32)
                eps_tensor = tf.constant(eps_value, name='eps', dtype=eps_value.dtype.type, shape=eps_value.shape)
                var_mv_sum_tensor = tf.add(var_mv_tensor, eps_tensor)
                gamma_mult_tensor_mv = tf.multiply(gamma_tensor, tf.math.rsqrt(var_mv_sum_tensor))
                mean_mult_tensor_mv = tf.multiply(mean_mv_tensor, gamma_mult_tensor_mv)
                var_b_sum_tensor = tf.add(unbessel_var_b_tensor, eps_tensor)
                gamma_mult_tensor_b = tf.multiply(gamma_tensor, tf.math.rsqrt(var_b_sum_tensor))
                mean_mult_tensor_b = tf.multiply(mean_b_tensor, gamma_mult_tensor_b)
                corr_recip_tensor = tf.add(freeze_bn_tensor, (1 - freeze_bn_tensor) * tf.sqrt(var_mv_sum_tensor / var_b_sum_tensor), name='corr_recip')
                corr_offset_tensor = tf.add(freeze_bn_tensor * mean_mult_tensor_mv, (1 - freeze_bn_tensor) * mean_mult_tensor_b, name='corr_offset')
                bias_tensor = tf.subtract(beta_tensor, corr_offset_tensor, name='biases')
                if layer_node.op == 'DepthwiseConv2dNative':
                    gamma_mult_tensor_mv = tf.reshape(gamma_mult_tensor_mv, [in_channels, depth_multiplier])
            with temp_graph.name_scope(scope + sep):
                weight_tensor = tf.compat.v1.placeholder(tf.float32)
                cloned_layer_tensor = tf.compat.v1.placeholder(tf.float32)
                scaled_weight_tensor = tf.multiply(weight_tensor, gamma_mult_tensor_mv, name='Mul_fold')
                scaled_layer_tensor = tf.multiply(cloned_layer_tensor, corr_recip_tensor, name='correction')
                bias_add_tensor = tf.nn.bias_add(scaled_layer_tensor, bias_tensor, name='Add_fold')
        replace_map = {}
        replace_map[layer_tensor.op.node_def.name] = layer_node.name
        replace_map[mean_b_tensor.op.node_def.name] = mean_b_node_name
        replace_map[var_b_tensor.op.node_def.name] = variance_b_node_name
        replace_map[mean_mv_tensor.op.node_def.name] = mean_mv_node.name
        replace_map[var_mv_tensor.op.node_def.name] = variance_mv_node.name
        replace_map[beta_tensor.op.node_def.name] = beta_node.name
        replace_map[gamma_tensor.op.node_def.name] = gamma_node.name
        replace_map[weight_tensor.op.node_def.name] = weight_node.name
        replace_map[cloned_layer_tensor.op.node_def.name] = cloned_layer_node.name
        replace_map[freeze_bn_tensor.op.node_def.name] = freeze_bn_node.name
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

        del cloned_layer_node.input[1]
        cloned_layer_node.input.extend([scaled_weight_tensor.op.node_def.name])
        for final_output_node_name, input_index in final_output_nodes.items():
            final_output_node = node_map[final_output_node_name]
            del final_output_node.input[input_index]
            final_output_node.input.insert(input_index, bias_add_tensor.op.node_def.name)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def _fold_fused_batch_norms_inference(input_graph_def):
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for match_result in graph_matcher.find_fused_batch_norms(input_graph_def):
        weight_node = match_result.get_op('weight_pattern')
        gamma_node = match_result.get_op('gamma_pattern')
        beta_node = match_result.get_op('beta_pattern')
        mean_mv_node = match_result.get_op('mean_pattern')
        variance_mv_node = match_result.get_op('variance_pattern')
        layer_node = match_result.get_op('layer_pattern')
        bn_node = match_result.get_op('batch_norm_pattern')
        final_output_nodes = output_node_map[bn_node.name]
        if bn_node.attr['is_training'].b is True:
            delete_keys = []
            for node_name, input_index in final_output_nodes.items():
                if 'AssignMovingAvg' in node_name:
                    moving_avg_sub_node = node_map[node_name]
                    moving_avg_node = graph_utils.node_from_map(node_map, moving_avg_sub_node.input[0])
                    if 'moving_mean' in moving_avg_node.name:
                        mean_mv_node = moving_avg_node
                        purge_mv_mean_scope = moving_avg_sub_node.name.rpartition('/')[0]
                    else:
                        if 'moving_variance' in moving_avg_node.name:
                            variance_mv_node = moving_avg_node
                            purge_mv_var_scope = moving_avg_sub_node.name.rpartition('/')[0]
                    delete_keys.append(node_name)

            for key in delete_keys:
                del final_output_nodes[key]

            for node in input_graph_def.node:
                if purge_mv_mean_scope in node.name or purge_mv_var_scope in node.name:
                    nodes_to_skip[node.name] = True

        if layer_node.op == 'MatMul':
            matmul_reshape_node = match_result.get_op('matmul_reshape_pattern')
            matmul_bn_output_reshape_node = match_result.get_op('matmul_bn_output_reshape_pattern')
            final_output_nodes = output_node_map[matmul_bn_output_reshape_node.name]
        nodes_to_skip[bn_node.name] = True
        if layer_node.op == 'MatMul':
            nodes_to_skip[matmul_reshape_node.name] = True
            nodes_to_skip[matmul_bn_output_reshape_node.name] = True
        if layer_node.op == 'DepthwiseConv2dNative':
            in_channels = weight_node.attr['_output_shapes'].list.shape[0].dim[2].size
            depth_multiplier = weight_node.attr['_output_shapes'].list.shape[0].dim[3].size
        temp_graph = tf.Graph()
        with temp_graph.as_default():
            scope, sep, _ = layer_node.name.rpartition('/')
            with temp_graph.name_scope(scope + sep + 'BatchNorm_Fold' + sep):
                mean_mv_tensor = tf.compat.v1.placeholder(tf.float32)
                var_mv_tensor = tf.compat.v1.placeholder(tf.float32)
                beta_tensor = tf.compat.v1.placeholder(tf.float32)
                gamma_tensor = tf.compat.v1.placeholder(tf.float32)
                eps_value = np.array(bn_node.attr['epsilon'].f, dtype=np.float32)
                eps_tensor = tf.constant(eps_value, name='eps', dtype=eps_value.dtype.type, shape=eps_value.shape)
                var_mv_sum_tensor = tf.add(var_mv_tensor, eps_tensor)
                gamma_mult_tensor = tf.multiply(gamma_tensor, tf.math.rsqrt(var_mv_sum_tensor))
                mean_mult_tensor = tf.multiply(mean_mv_tensor, gamma_mult_tensor)
                bias_tensor = tf.subtract(beta_tensor, mean_mult_tensor, name='biases')
                if layer_node.op == 'DepthwiseConv2dNative':
                    gamma_mult_tensor = tf.reshape(gamma_mult_tensor, [in_channels, depth_multiplier])
            with temp_graph.name_scope(scope + sep):
                weight_tensor = tf.compat.v1.placeholder(tf.float32)
                layer_tensor = tf.compat.v1.placeholder(tf.float32)
                scaled_weight_tensor = tf.multiply(weight_tensor, gamma_mult_tensor, name='Mul_fold')
                bias_add_tensor = tf.nn.bias_add(layer_tensor, bias_tensor, name='Add_fold')
        replace_map = {}
        replace_map[mean_mv_tensor.op.node_def.name] = mean_mv_node.name
        replace_map[var_mv_tensor.op.node_def.name] = variance_mv_node.name
        replace_map[beta_tensor.op.node_def.name] = beta_node.name
        replace_map[gamma_tensor.op.node_def.name] = gamma_node.name
        replace_map[weight_tensor.op.node_def.name] = weight_node.name
        replace_map[layer_tensor.op.node_def.name] = layer_node.name
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

        del layer_node.input[1]
        layer_node.input.extend([scaled_weight_tensor.op.node_def.name])
        for final_output_node_name, input_index in final_output_nodes.items():
            final_output_node = node_map[final_output_node_name]
            del final_output_node.input[input_index]
            final_output_node.input.insert(input_index, bias_add_tensor.op.node_def.name)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def