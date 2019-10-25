#! /usr/bin/env python 3.6 (3379)
#coding=utf-8
# Compiled at: 2019-10-23 20:32:36
#Powered by BugScaner
#http://tools.bugscaner.com/
#如果觉得不错,请分享给你朋友使用吧!
"""
Quantize graph utils

@ author: Sambhav Jain
"""
import numpy as np, collections, os, re, json, h5py, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from graffitist.utils import graph_utils, graph_matcher
kernel_root = os.path.abspath(os.path.join(__file__, '../../kernels'))
if tf.test.is_built_with_cuda():
    if tf.test.is_gpu_available(cuda_only=True):
        _quantize_ops_module = tf.load_op_library(os.path.join(kernel_root, 'quantize_ops_cuda.so'))
    _quantize_ops_module = tf.load_op_library(os.path.join(kernel_root, 'quantize_ops.so'))
linear_quant_kernel = _quantize_ops_module.linear_quant
linear_quant_grad_kernel = _quantize_ops_module.linear_quant_gradient

@tf.RegisterGradient('LinearQuant')
def _linear_quant_grad(op, grad):
    grad_wrt_inputs, grad_wrt_scale = linear_quant_grad_kernel(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.get_attr('rounding_mode'))
    return (
     grad_wrt_inputs, grad_wrt_scale, None, None)


def quantize_layers(input_graph_def, weight_bits, activation_bits, layer_bits, relu_bits, first_layer, last_layer, is_training):
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for match_result in graph_matcher.find_layers_to_quantize(input_graph_def):
        input_node = match_result.get_op('input_pattern')
        weight_node = match_result.get_op('weight_cast_pattern') or match_result.get_op('weight_identity_pattern') or match_result.get_op('weight_var_pattern') or match_result.get_op('weight_resource_var_pattern') or match_result.get_op('frozen_weight_pattern') or match_result.get_op('folded_weight_pattern')
        layer_node = match_result.get_op('layer_pattern')
        bn_correction_node = match_result.get_op('bn_correction_pattern')
        bias_node = match_result.get_op('bias_pattern')
        bias_add_node = match_result.get_op('bias_add_pattern')
        bypass_node = match_result.get_op('bypass_pattern')
        relu_node = match_result.get_op('activation_relu_pattern')
        leakyrelu_node = match_result.get_op('activation_leakyrelu_pattern')
        leakyrelu_alpha_node = match_result.get_op('leaky_relu_alpha_pattern')
        leakyrelu_alpha_x_node = match_result.get_op('leaky_relu_alpha_x_pattern')
        if 'gru' in layer_node.name:
            continue
        dtype = tf.float32
        if input_node.op == 'Cast' and input_node.attr['DstT'].type == 2:
            dtype = tf.float64
        if first_layer:
            if first_layer.rpartition('/')[0] == layer_node.name.rpartition('/')[0]:
                _insert_quant_op('weight_quant', weight_node, is_training, -8, new_nodes, dtype=dtype)
            if last_layer:
                if last_layer.rpartition('/')[0] == layer_node.name.rpartition('/')[0]:
                    _insert_quant_op('weight_quant', weight_node, is_training, -8, new_nodes, dtype=dtype)
                _insert_quant_op('weight_quant', weight_node, is_training, weight_bits, new_nodes, dtype=dtype)
        if bn_correction_node is not None:
            if not is_training:
                raise ValueError("Invalid setting. Use is_training=True for 'quantize' transform as done for 'fold_batch_norms' transform.")
            _insert_quant_op('layer_quant', bn_correction_node, is_training, layer_bits, new_nodes, dtype=dtype)
        else:
            _insert_quant_op('layer_quant', layer_node, is_training, layer_bits, new_nodes, dtype=dtype)
        _insert_quant_op('bias_quant', bias_node, is_training, layer_bits, new_nodes, dtype=dtype)
        if bypass_node is None:
            if relu_node is None:
                if leakyrelu_node is None:
                    _insert_quant_op('biasadd_quant', bias_add_node, is_training, activation_bits, new_nodes, dtype=dtype)
        if bypass_node is not None:
            _insert_quant_op('biasadd_quant', bias_add_node, is_training, activation_bits, new_nodes, dtype=dtype)
        if bypass_node is not None:
            if relu_node is None:
                _insert_quant_op('eltwadd_quant', bypass_node, is_training, activation_bits, new_nodes, dtype=dtype)
        if relu_node is not None:
            _insert_quant_op('act_quant', relu_node, is_training, relu_bits, new_nodes, dtype=dtype)
        if leakyrelu_node is not None:
            _insert_quant_op('biasadd_quant', bias_add_node, is_training, layer_bits, new_nodes, dtype=dtype)
            _insert_quant_op('lrelu_weight_quant', leakyrelu_alpha_node, is_training, layer_bits, new_nodes, dtype=dtype)
            _insert_quant_op('lrelu_alpha_x_quant', leakyrelu_alpha_x_node, is_training, layer_bits, new_nodes, dtype=dtype)
            _insert_quant_op('act_quant', leakyrelu_node, is_training, activation_bits, new_nodes, dtype=dtype)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def quantize_input(input_graph_def, input_node_names, activation_bits, is_training):
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for input_node_name in input_node_names:
        if ':' in input_node_name:
            raise ValueError("Name '%s' appears to refer to a Tensor, not a Operation." % input_node_name)
        input_node = graph_utils.node_from_map(node_map, input_node_name)
        _insert_quant_op('inp_quant', input_node, is_training, activation_bits, new_nodes)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def quantize_separable_conv(input_graph_def, weight_bits, activation_bits, first_layer, last_layer, is_training):
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for node in input_graph_def.node:
        if node.op == 'Conv2D':
            input_node = graph_utils.node_from_map(node_map, node.input[0])
            if input_node.op == 'DepthwiseConv2dNative':
                pass
        if '_from_avgpool' not in input_node.name:
            layer_node = input_node
            weight_node = graph_utils.node_from_map(node_map, layer_node.input[1])
            if first_layer:
                if first_layer.rpartition('/')[0] == layer_node.name.rpartition('/')[0]:
                    _insert_quant_op('weight_quant', weight_node, is_training, -8, new_nodes)
                if last_layer:
                    if last_layer.rpartition('/')[0] == layer_node.name.rpartition('/')[0]:
                        _insert_quant_op('weight_quant', weight_node, is_training, -8, new_nodes)
                    _insert_quant_op('weight_quant', weight_node, is_training, weight_bits, new_nodes)
            _insert_quant_op('layer_quant', layer_node, is_training, activation_bits, new_nodes)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def quantize_rescale(input_graph_def, activation_bits, relu_bits, is_training):
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for node in input_graph_def.node:
        if node.op == 'Identity':
            if 'eltw_rescale' in node.name:
                _insert_quant_op('eltw_rescale_quant', node, is_training, activation_bits, new_nodes)
        elif node.op == 'Identity' and 'concat_rescale' in node.name:
            consumer_nodes = output_node_map[node.name]
            for consumer_node_name, input_index in consumer_nodes.items():
                concat_node = node_map[consumer_node_name]

            if not concat_node.op == 'ConcatV2':
                raise AssertionError
            for input_node_name in concat_node.input:
                if 'LinearQuant' in input_node_name:
                    quant_issigned_node_name = input_node_name.replace('LinearQuant', 'is_signed')
                    quant_issigned_node = graph_utils.node_from_map(node_map, quant_issigned_node_name)
                    is_signed_value = int(graph_utils.values_from_const(quant_issigned_node))

            if is_signed_value == 1:
                _insert_quant_op('concat_rescale_quant', node, is_training, activation_bits, new_nodes)
            else:
                _insert_quant_op('concat_rescale_quant', node, is_training, relu_bits, new_nodes)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def quantize_avgpool(input_graph_def, avgpool_bits, avgpool_reciprocal_bits, is_training):
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for node in input_graph_def.node:
        if node.op == 'AvgPool':
            _insert_quant_op('avgpool_quant', node, is_training, avgpool_bits, new_nodes)
        elif node.op == 'DepthwiseConv2dNative':
            if '_from_avgpool' in node.name:
                avgpool_reciprocal_node = graph_utils.node_from_map(node_map, node.input[1])
                _insert_quant_op('weight_quant', avgpool_reciprocal_node, is_training, avgpool_reciprocal_bits, new_nodes, dtype=tf.float32)
                _insert_quant_op('avgpool_quant', node, is_training, avgpool_bits, new_nodes, rounding_mode='floor', dtype=tf.float32)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def _insert_quant_op(quant_name, producer_node, is_training, bitwidth, new_nodes, rounding_mode='round', dtype=tf.float32):
    if producer_node.op == 'Identity':
        if 'read' in producer_node.name:
            producer_node_name = graph_utils.node_name_from_input(producer_node.input[0])
        if is_training:
            if producer_node.name.split('/')[-1] == 'correction':
                producer_node_name = graph_utils.node_name_from_input(producer_node.input[0]).replace('_Fold', '')
            producer_node_name = producer_node.name
    scope, sep, name = producer_node_name.rpartition('/')
    temp_graph = tf.Graph()
    with temp_graph.as_default():
        with temp_graph.name_scope(scope + sep + name + '_' + quant_name + sep):
            input_tensor = tf.compat.v1.placeholder(dtype)
            output_tensor = _linear_quant_v2(input_tensor, is_training, bitwidth, rounding_mode, dtype)
    replace_map = {}
    replace_map[input_tensor.op.node_def.name] = producer_node.name
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

    consumer_nodes = output_node_map[producer_node.name]
    for consumer_node_name, input_index in consumer_nodes.items():
        consumer_node = node_map[consumer_node_name]
        del consumer_node.input[input_index]
        consumer_node.input.insert(input_index, output_tensor.op.node_def.name)


def _linear_quant_v2(input, is_training, bitwidth, rounding_mode='round', dtype=tf.float32, precalibrated_threshold=None):
    if precalibrated_threshold is None:
        with tf.compat.v1.variable_scope(tf.compat.v1.get_default_graph().get_name_scope()):
            log2_t = tf.compat.v1.get_variable('threshold', [], dtype=dtype, initializer=tf.constant_initializer(0.0))
    else:
        log2_t = tf.constant(np.log2(precalibrated_threshold), dtype, name='pc_threshold')
    if bitwidth < 0:
        is_signed = tf.constant(1.0, dtype, name='is_signed')
    else:
        is_signed = tf.constant(0.0, dtype, name='is_signed')
    bw = tf.constant(abs(bitwidth), dtype, name='bitwidth')
    base = tf.constant(2.0, dtype, name='base')
    bits = tf.subtract(bw, is_signed, name='bits')
    q_levels_per_sign = tf.pow(base, bits, name='q_levels_per_sign')
    neg_limit = -q_levels_per_sign * is_signed
    pos_limit = q_levels_per_sign - 1.0
    log2_t = tf.identity(log2_t, name='pof2')
    if is_training:
        freeze_th_default = tf.constant(False, dtype=tf.bool, shape=[], name='freeze_th_default')
        freeze_th_bool = tf.compat.v1.placeholder_with_default(freeze_th_default, shape=[], name='freeze_th')
        freeze_th = tf.cast(freeze_th_bool, dtype, name='freeze_th_cast')
        log2_t = freeze_th * tf.stop_gradient(log2_t) + (1.0 - freeze_th) * log2_t
        ceil_log2_t = log2_t + tf.stop_gradient(tf.math.ceil(log2_t) - log2_t)
    else:
        ceil_log2_t = tf.math.ceil(log2_t)
    threshold = tf.pow(base, ceil_log2_t, name='pof2_t')
    scaling_factor = tf.divide(threshold, q_levels_per_sign, name='pof2_sf')
    if rounding_mode not in ('round', 'floor', 'ceil'):
        raise ValueError('Invalid rounding mode selected: %s' % rounding_mode)
    output = linear_quant_kernel(input, scaling_factor, neg_limit, pos_limit, rounding_mode)
    return output


def auto_merge_quant_layers(input_graph_def):
    r"""
    This function automatically merges certain quant layers (horizontally)
    to share the same quantization parameters (scale factors).
    
    For instance, Concat's input activations need to have the same quantization
    parameters (scale factors) to avoid the need for rescaling. Otherwise rescaling 
    will make Concat a lossy operation.
      
        act_quant_1  act_quant_2  act_quant_3  act_quant_4
             |            |            |            |
             \            \            /            /
                              Concat
    
    Also applies if any of the branches do not contain act_quant (e.g. maxpool
    in inception v4's mixed_6a).
    
        act_quant_1  act_quant_2  act_quant_3  concat_rescale_quant
             |            |            |            |
             \            \            /            /
                              Concat
    
    Another case is inputs to BiasAdd, which also need to have the same quantization
    parameters (scale factors).
    
                     layer_quant   bias_quant
                          |            |
                          \            /
                              BiasAdd
    
    Yet another instance is inputs to EltwiseAdd (resnet-v1 bypass/shortcut connections).
    This has two cases: 1) Projection shortcuts; 2) Identity shortcuts.
    
    Case 1) Projection Shortcuts
    
                  biasadd_quant   biasadd_quant
                          |            |
                          \            /
                            EltwiseAdd
    
    Case 2) Identity Shortcuts
    
                  biasadd_quant   eltw_rescale_quant
                          |            |
                          \            /
                            EltwiseAdd
    
    Known limitation: Only supports 2 input eltwise add structures for now.
    
    Yet yet another case is Leaky ReLU which is implemented as y = max(alpha*x, x). Here
    the two inputs of Maximum op should have the same scale.
    
              lrelu_alpha_x_quant    biasadd_quant
                          |            |
                          \            /
                             Maximum
    
    There could be other cases where quant layers need to be merged, such as when
    fusing operations horizontally or vertically. This can be implemented by providing
    a manual list of quant layers to be merged to the function manual_merge_quant_layers.
    
    Always run manual_merge_quant_layers AFTER auto_merge_quant_layers.
    """
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for node in input_graph_def.node:
        node_suffix = node.name.rpartition('/')[-1]
        if node.op == 'ConcatV2':
            _merge_quant_before_this_node(node_suffix + '_act_quant_concat_merge', node, nodes_to_skip, new_nodes)
        elif node.op == 'Maximum':
            if 'biasadd_quant' in node.input[0] and 'lrelu_alpha_x_quant' in node.input[1] or 'biasadd_quant' in node.input[1] and 'lrelu_alpha_x_quant' in node.input[0]:
                _merge_quant_before_this_node(node_suffix + '_leakyrelu_quant_merge', node, nodes_to_skip, new_nodes)
        elif node.op == 'Add' or node.op == 'BiasAdd':
            if 'layer_quant' in node.input[0] and 'bias_quant' in node.input[1] or 'layer_quant' in node.input[1] and 'bias_quant' in node.input[0]:
                _merge_quant_before_this_node(node_suffix + '_layer_bias_quant_merge', node, nodes_to_skip, new_nodes)
        elif 'biasadd_quant' in node.input[0]:
            if 'biasadd_quant' in node.input[1]:
                _merge_quant_before_this_node(node_suffix + '_eltw_biasadd_quant_merge', node, nodes_to_skip, new_nodes)
        elif 'biasadd_quant' in node.input[0] and 'eltw_rescale_quant' in node.input[1] or 'biasadd_quant' in node.input[1] and 'eltw_rescale_quant' in node.input[0]:
            _merge_quant_before_this_node(node_suffix + '_eltw_rescale_quant_merge', node, nodes_to_skip, new_nodes)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def _merge_quant_before_this_node(name, node, nodes_to_skip, new_nodes):
    quant_threshold_nodes = []
    merged_quant_type = set()
    merged_bitwidths = set()
    merged_sign_bits = set()
    merged_dtypes = set()
    for input_name in node.input:
        input_node = graph_utils.node_from_map(node_map, input_name)
        if '_quant/LinearQuant' in input_node.name:
            quant_threshold_node_name = input_node.name.replace('LinearQuant', 'threshold')
            quant_threshold_nodes.append(quant_threshold_node_name)
            if input_node.name.replace('LinearQuant', 'pof2_sf') in node_map:
                merged_quant_type.add('pof2')
            else:
                merged_quant_type.add('non_pof2')
            quant_bitwidth_node_name = input_node.name.replace('LinearQuant', 'bitwidth')
            quant_bitwidth_node = graph_utils.node_from_map(node_map, quant_bitwidth_node_name)
            bitwidth_value = int(graph_utils.values_from_const(quant_bitwidth_node))
            merged_bitwidths.add(bitwidth_value)
            quant_issigned_node_name = input_node.name.replace('LinearQuant', 'is_signed')
            quant_issigned_node = graph_utils.node_from_map(node_map, quant_issigned_node_name)
            is_signed_value = int(graph_utils.values_from_const(quant_issigned_node))
            merged_sign_bits.add(is_signed_value)
            dtype_enum = quant_issigned_node.attr['dtype'].type
            merged_dtypes.add(dtype_enum)

    if len(quant_threshold_nodes) != 0:
        if len(merged_quant_type) != 1:
            print('Quant nodes:', quant_threshold_nodes)
            print('Types:', merged_quant_type)
            raise ValueError("The quant layers being merged do not have the same type. Can't be merged!")
        if len(merged_bitwidths) != 1:
            print('Quant nodes:', quant_threshold_nodes)
            print('Bitwidths:', merged_bitwidths)
            raise ValueError("The quant layers being merged do not share the same bitwidth. Can't be merged!")
        if len(merged_sign_bits) != 1:
            print('Quant nodes:', quant_threshold_nodes)
            print('Sign bits:', merged_sign_bits)
            raise ValueError("The quant layers being merged do not share the same sign bit. Can't be merged!")
        if len(merged_dtypes) != 1:
            print('Quant nodes:', quant_threshold_nodes)
            print('Datatypes (enum):', merged_dtypes)
            raise ValueError("The quant layers being merged do not share the same datatype. Can't be merged!")
        common_prefix = os.path.commonprefix(list(quant_threshold_nodes))
        if common_prefix == '':
            common_prefix = quant_threshold_nodes[0]
        scope, sep, _ = common_prefix.rpartition('/')
        if scope == '':
            scope = common_prefix
            sep = '/'
        temp_graph = tf.Graph()
        with temp_graph.as_default():
            with temp_graph.name_scope(scope + sep + name + sep):
                with tf.compat.v1.variable_scope(tf.compat.v1.get_default_graph().get_name_scope()):
                    threshold_tensor = tf.compat.v1.get_variable('m_threshold', [], dtype=merged_dtypes.pop(), initializer=tf.constant_initializer(0.0))
        temp_graph_def = temp_graph.as_graph_def(add_shapes=True)
        for node in temp_graph_def.node:
            new_nodes.extend([node])

        for old_threshold_node_name in quant_threshold_nodes:
            nodes_to_skip[old_threshold_node_name] = True
            nodes_to_skip[old_threshold_node_name + '/read'] = True
            nodes_to_skip[old_threshold_node_name + '/Assign'] = True
            nodes_to_skip[old_threshold_node_name + '/Initializer/Const'] = True
            consumer_nodes = output_node_map[old_threshold_node_name + '/read']
            for consumer_node_name, input_index in consumer_nodes.items():
                consumer_node = node_map[consumer_node_name]
                del consumer_node.input[input_index]
                consumer_node.input.insert(input_index, threshold_tensor.op.node_def.name + '/read')


def manual_merge_quant_layers(input_graph_def, layer_merge_list):
    """
    This function manually merges quant layers following the nodes provided in layer_merge_list,
    to share the same quantization parameters (scale factors). This is useful when fusing operations 
    horizontally or vertically. The nodes remain physically separate, but by sharing the quantization 
    thresholds it numerically models the nodes being actually merged in implementation.
    
    layer_merge_list is the path to a .txt file containing node names to be merged (comma separated).
    
    E.g.
    
      inception_3a_3x3_reduce/weights, inception_3a_5x5_reduce/weights
      inception_3a_3x3_reduce/biases, inception_3a_5x5_reduce/biases
      inception_3a_3x3_reduce/Conv2D, inception_3a_5x5_reduce/Conv2D
      inception_3a_3x3_reduce/BiasAdd, inception_3a_5x5_reduce/BiasAdd
      inception_3a_3x3_reduce/inception_3a_3x3_reduce, inception_3a_5x5_reduce/inception_3a_5x5_reduce
    
    Always run manual_merge_quant_layers AFTER auto_merge_quant_layers.
    """
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    merge_list = []
    with open(layer_merge_list, 'r') as (f):
        for line in f.readlines():
            if line != '\n':
                merge_list.append(line.strip().split(', '))

    for merge_nodes_set in merge_list:
        quant_layer_set = []
        for merge_node in merge_nodes_set:
            consumer_nodes = output_node_map[merge_node]
            for consumer_node_name, input_index in consumer_nodes.items():
                pass

            quant_scale_node = node_map[consumer_node_name]
            quant_layer, _, _ = quant_scale_node.name.rpartition('/')
            if '_quant' in quant_layer:
                quant_layer_set.append(quant_layer)

        if len(quant_layer_set) == 1:
            raise ValueError(("Unable to find a quant layer to merge '{}' with!").format(quant_layer_set[0]))
        elif len(quant_layer_set) > 1:
            if 'weight_quant' in quant_layer_set[0]:
                _merge_quant_layers_in_set('weight_quant_manual_merge', quant_layer_set, nodes_to_skip, new_nodes)
            elif 'bias_quant' in quant_layer_set[0]:
                _merge_quant_layers_in_set('bias_quant_manual_merge', quant_layer_set, nodes_to_skip, new_nodes)
            elif 'layer_quant' in quant_layer_set[0]:
                _merge_quant_layers_in_set('layer_quant_manual_merge', quant_layer_set, nodes_to_skip, new_nodes)
            elif 'act_quant' in quant_layer_set[0]:
                _merge_quant_layers_in_set('act_quant_manual_merge', quant_layer_set, nodes_to_skip, new_nodes)
            else:
                raise ValueError(("Quant layer name '{}' inferred from layer_merge_list is unknown!").format(quant_layer_set[0]))

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def _merge_quant_layers_in_set(name, quant_layer_set, nodes_to_skip, new_nodes):
    quant_threshold_nodes = set()
    merged_quant_type = set()
    merged_bitwidths = set()
    merged_sign_bits = set()
    merged_dtypes = set()
    for quant_scope in quant_layer_set:
        if quant_scope + '/pof2_sf' in node_map:
            merged_quant_type.add('pof2')
            threshold_consumer_node_name = quant_scope + '/pof2'
        else:
            merged_quant_type.add('non_pof2')
            threshold_consumer_node_name = quant_scope + '/nonpof2'
        threshold_consumer_node = graph_utils.node_from_map(node_map, threshold_consumer_node_name)
        quant_threshold_node_name = threshold_consumer_node.input[0].replace('/read', '')
        quant_threshold_nodes.add(quant_threshold_node_name)
        quant_bitwidth_node_name = quant_scope + '/bitwidth'
        quant_bitwidth_node = graph_utils.node_from_map(node_map, quant_bitwidth_node_name)
        bitwidth_value = int(graph_utils.values_from_const(quant_bitwidth_node))
        merged_bitwidths.add(bitwidth_value)
        quant_issigned_node_name = quant_scope + '/is_signed'
        quant_issigned_node = graph_utils.node_from_map(node_map, quant_issigned_node_name)
        is_signed_value = int(graph_utils.values_from_const(quant_issigned_node))
        merged_sign_bits.add(is_signed_value)
        dtype_enum = quant_issigned_node.attr['dtype'].type
        merged_dtypes.add(dtype_enum)

    if len(quant_threshold_nodes) > 1:
        if len(merged_quant_type) != 1:
            print('Quant nodes:', quant_threshold_nodes)
            print('Types:', merged_quant_type)
            raise ValueError("The quant layers being merged do not have the same type. Can't be merged!")
        if len(merged_bitwidths) != 1:
            print('Quant nodes:', quant_threshold_nodes)
            print('Bitwidths:', merged_bitwidths)
            raise ValueError("The quant layers being merged do not share the same bitwidth. Can't be merged!")
        if len(merged_sign_bits) != 1:
            print('Quant nodes:', quant_threshold_nodes)
            print('Sign bits:', merged_sign_bits)
            raise ValueError("The quant layers being merged do not share the same sign bit. Can't be merged!")
        if len(merged_dtypes) != 1:
            print('Quant nodes:', quant_threshold_nodes)
            print('Datatypes (enum):', merged_dtypes)
            raise ValueError("The quant layers being merged do not share the same datatype. Can't be merged!")
        common_prefix = os.path.commonprefix(list(quant_threshold_nodes))
        if common_prefix == '':
            common_prefix = quant_threshold_nodes[0]
        scope, sep, _ = common_prefix.rpartition('/')
        if scope == '':
            scope = common_prefix
            sep = '/'
        temp_graph = tf.Graph()
        with temp_graph.as_default():
            with temp_graph.name_scope(scope + sep + name + sep):
                with tf.compat.v1.variable_scope(tf.compat.v1.get_default_graph().get_name_scope()):
                    threshold_tensor = tf.compat.v1.get_variable('m_threshold', [], dtype=merged_dtypes.pop(), initializer=tf.constant_initializer(0.0))
        temp_graph_def = temp_graph.as_graph_def(add_shapes=True)
        for node in temp_graph_def.node:
            new_nodes.extend([node])

        for old_threshold_node_name in quant_threshold_nodes:
            nodes_to_skip[old_threshold_node_name] = True
            nodes_to_skip[old_threshold_node_name + '/read'] = True
            nodes_to_skip[old_threshold_node_name + '/Assign'] = True
            nodes_to_skip[old_threshold_node_name + '/Initializer/Const'] = True
            consumer_nodes = output_node_map[old_threshold_node_name + '/read']
            for consumer_node_name, input_index in consumer_nodes.items():
                consumer_node = node_map[consumer_node_name]
                del consumer_node.input[input_index]
                consumer_node.input.insert(input_index, threshold_tensor.op.node_def.name + '/read')


def calibrate_quant_layers(input_graph_def, input_node_names, ckpt_path, calib_path, tf_collections, is_training, verbose):
    r"""
    Note: The nodes in input_graph_def.node are not arranged in any particular order,
    certainly not in the topological order (one in which nodes are executed such that 
    the dependencies are resolved).
    
    E.g. consider this DAG (directed-acyclic-graph), with data flowing down:
    
               a
               |
               b
              /            c   d
             |   |
             |   e
              \ /
               f
    
    DFS (depth-first-search) traversals: 
    a-b-c-f-d-e   :(   f depends on d and e
    a-b-d-e-f-c   :(   f depends on c
    
    BFS (breadth-first-search) traversals:
    a-b-c-d-f-e   :(   f depends on e
    a-b-d-c-e-f   :)
    
    Topologically sorted orders: 
    a-b-c-d-e-f   :)
    a-b-d-e-c-f   :)
    a-b-d-c-e-f   :)
    
    When calibrating the quant layers, it is important to do it in a topological order, as the thresholds
    of consuming layers will depend on how the producing layers have been thresholded.
    """
    global node_map
    global output_node_map
    sorted_node_names = graph_utils.sort_graph_topological(input_graph_def)
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    input_features = np.load(calib_path, encoding='latin1')
    merged_quant_calib_status = collections.defaultdict(set)
    merged_quant_input_data = collections.defaultdict()
    if is_training:
        if '_graffitist' in ckpt_path:
            raise ValueError(("Thresholds already calibrated by Graffitist. Please use original ckpt when is_training=True instead of '{}'.").format(ckpt_path))
    with tf.compat.v1.Session(graph=tf.Graph()) as (sess):
        tf.import_graph_def(input_graph_def, name='')
        var_list = {}
        if ckpt_path.endswith('.ckpt'):
            reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
            for key in reader.get_variable_to_shape_map():
                try:
                    tensor = sess.graph.get_tensor_by_name(key + ':0')
                except KeyError:
                    continue

                var_list[key] = tensor

            saver = tf.compat.v1.train.Saver(var_list=var_list)
            saver.restore(sess, ckpt_path)
        else:
            ckpt_path = os.path.join(ckpt_path, 'frozen_model.ckpt')
        g = tf.compat.v1.get_default_graph()
        global_vars, threshold_vars = {}, {}
        if is_training:
            if len(tf_collections) != 0:
                for node in input_graph_def.node:
                    if 'Variable' in node.op:
                        if node.name + ':0' in tf_collections[tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES]['variables']:
                            global_vars[node.name] = graph_utils.restored_variable(node.name, trainable=True)
                        else:
                            if node.name + ':0' in tf_collections[tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]['variables']:
                                global_vars[node.name] = graph_utils.restored_variable(node.name, trainable=False)
                            else:
                                continue

                for key in tf_collections.keys():
                    if len(tf_collections[key]['variables']) != 0:
                        if key != tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES:
                            if key != tf.compat.v1.GraphKeys.GLOBAL_VARIABLES:
                                for node_name in tf_collections[key]['variables']:
                                    node_name = graph_utils.node_name_from_input(node_name)
                                    if node_name in global_vars:
                                        g.add_to_collection(key, global_vars[node_name])

                    if len(tf_collections[key]['operations']) != 0:
                        for node_name in tf_collections[key]['operations']:
                            g.add_to_collection(key, g.get_operation_by_name(node_name))

                    if len(tf_collections[key]['tensors']) != 0:
                        for node_name in tf_collections[key]['tensors']:
                            g.add_to_collection(key, g.get_tensor_by_name(node_name))

        for node_name in sorted_node_names:
            if '_quant/LinearQuant' in node_name:
                quant_scope = re.match('.*_quant/', node_name).group(0)
                quant_node = graph_utils.node_from_map(node_map, node_name)
                quant_input_node = graph_utils.node_from_map(node_map, quant_node.input[0])
                quant_bitwidth_node_name = quant_scope + 'bitwidth'
                quant_bitwidth_node = graph_utils.node_from_map(node_map, quant_bitwidth_node_name)
                bitwidth_value = int(graph_utils.values_from_const(quant_bitwidth_node))
                quant_issigned_node_name = quant_scope + 'is_signed'
                quant_issigned_node = graph_utils.node_from_map(node_map, quant_issigned_node_name)
                is_signed_value = int(graph_utils.values_from_const(quant_issigned_node))
                quant_threshold_node_name = quant_scope + 'threshold'
                if quant_scope + 'pof2' in node_map:
                    scaling_mode = 'pof2'
                else:
                    raise ValueError('Quant type unknown - not using pof2 scaling!')
                if quant_threshold_node_name in node_map:
                    quant_threshold_node = graph_utils.node_from_map(node_map, quant_threshold_node_name)
                else:
                    if scaling_mode == 'pof2':
                        quant_consumer_node = graph_utils.node_from_map(node_map, quant_scope + 'pof2')
                        quant_threshold_node = graph_utils.node_from_map(node_map, quant_consumer_node.input[0].replace('/read', ''))
                for input_node_name in input_node_names:
                    if ':' in input_node_name:
                        raise ValueError("Name '%s' appears to refer to a Tensor, not a Operation." % input_node_name)

                input = g.get_tensor_by_name(input_node_name + ':0')
                quant_input_tensor = g.get_tensor_by_name(quant_input_node.name + ':0')
                try:
                    quant_input_data = sess.run(quant_input_tensor, {input: input_features})
                except:
                    batch_size = input_features.shape[0]
                    mini_batch_size = batch_size
                    while 1:
                        mini_batch_size = mini_batch_size // 2
                        quant_input_data = None
                        try:
                            for i in range(batch_size // mini_batch_size):
                                output = sess.run(quant_input_tensor, {input: input_features[mini_batch_size * i:mini_batch_size * (i + 1)]})
                                if quant_input_data is None:
                                    quant_input_data = output
                                else:
                                    quant_input_data = np.concatenate((quant_input_data, output), axis=0)

                            break
                        except:
                            continue

                quant_input_data = quant_input_data.ravel()
                if quant_threshold_node_name in node_map:
                    if 'weight_quant' in node_name:
                        if is_training:
                            threshold_value = np.abs(np.mean(quant_input_data)) + 3 * np.std(quant_input_data)
                        else:
                            threshold_value = np.max(np.abs(quant_input_data))
                    else:
                        threshold_value = _compute_threshold(quant_input_data, bitwidth_value, is_signed_value)
                    log2_t_value = np.log2(threshold_value)
                    if quant_threshold_node.name not in threshold_vars.keys():
                        var_list[quant_threshold_node.name] = g.get_tensor_by_name(quant_threshold_node.name + ':0')
                        threshold_vars[quant_threshold_node.name] = graph_utils.restored_variable(quant_threshold_node.name, trainable=True)
                    threshold_vars[quant_threshold_node.name].load(log2_t_value, sess)
                if scaling_mode == 'pof2':
                    scale = np.power(2.0, np.ceil(log2_t_value)) / np.power(2.0, bitwidth_value - is_signed_value)
                    pof2_scale = int(np.log2(scale))
                    if verbose:
                        print(('threshold: {:30.20f}   bitwidth: {:2d}   is_signed: {:1d}   scale: 2^{:<4d}   {}').format(threshold_value, bitwidth_value, is_signed_value, pof2_scale, quant_threshold_node.name))
                    else:
                        print(('calibrating:   {}').format(quant_threshold_node.name))
            else:
                if len(merged_quant_calib_status[quant_threshold_node.name]) == 0:
                    consumer_nodes = output_node_map[quant_threshold_node.name + '/read']
                    for consumer_node_name, _ in consumer_nodes.items():
                        consumer_scope = re.match('.*_quant/', consumer_node_name).group(0)
                        merged_quant_calib_status[quant_threshold_node.name].add(consumer_scope)

                    merged_quant_input_data[quant_threshold_node.name] = quant_input_data
                else:
                    merged_quant_input_data[quant_threshold_node.name] = np.concatenate((
                     merged_quant_input_data[quant_threshold_node.name], quant_input_data),
                      axis=0)
                merged_quant_calib_status[quant_threshold_node.name].remove(quant_scope)
                if 'weight_quant' in node_name:
                    if is_training:
                        threshold_value = np.abs(np.mean(merged_quant_input_data[quant_threshold_node.name])) + 3 * np.std(merged_quant_input_data[quant_threshold_node.name])
                    else:
                        threshold_value = np.max(np.abs(merged_quant_input_data[quant_threshold_node.name]))
                else:
                    threshold_value = _compute_threshold(merged_quant_input_data[quant_threshold_node.name], bitwidth_value, is_signed_value)
                log2_t_value = np.log2(threshold_value)
                if quant_threshold_node.name not in threshold_vars.keys():
                    var_list[quant_threshold_node.name] = g.get_tensor_by_name(quant_threshold_node.name + ':0')
                    threshold_vars[quant_threshold_node.name] = graph_utils.restored_variable(quant_threshold_node.name, trainable=True)
                threshold_vars[quant_threshold_node.name].load(log2_t_value, sess)
                if len(merged_quant_calib_status[quant_threshold_node.name]) == 0:
                    del merged_quant_input_data[quant_threshold_node.name]
                    if scaling_mode == 'pof2':
                        scale = np.power(2.0, np.ceil(log2_t_value)) / np.power(2.0, bitwidth_value - is_signed_value)
                        pof2_scale = int(np.log2(scale))
                        if verbose:
                            print(('threshold: {:30.20f}   bitwidth: {:2d}   is_signed: {:1d}   scale: 2^{:<4d}   {}').format(threshold_value, bitwidth_value, is_signed_value, pof2_scale, quant_threshold_node.name))
                        else:
                            print(('calibrating:   {}').format(quant_threshold_node.name))

        if '_graffitist' not in ckpt_path:
            if ckpt_path.endswith('.ckpt'):
                ckpt_path = ckpt_path.replace('.ckpt', '_graffitist.ckpt')
            else:
                ckpt_path += '_graffitist'
            saver = tf.compat.v1.train.Saver(var_list=var_list)
            saver.save(sess, ckpt_path, write_meta_graph=False)
            print(("Saved weights and calibrated thresholds to '{}'").format(ckpt_path))
            if is_training:
                if len(tf_collections) != 0:
                    saver.export_meta_graph(ckpt_path + '.meta', clear_devices=True, clear_extraneous_savers=True)
                    print(("Saved metagraph to '{}'").format(ckpt_path + '.meta'))
    return input_graph_def


def dump_quant_params(input_graph_def, ckpt_path, json_path, weights_path):
    """
    Dumps quantized weights and quantization parameters for activations.
    """
    global node_map
    global output_node_map
    sorted_node_names = graph_utils.sort_graph_topological(input_graph_def)
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    quant_params = {}
    quant_weights = h5py.File(weights_path, 'w')
    with tf.compat.v1.Session(graph=tf.Graph()) as (sess):
        tf.import_graph_def(input_graph_def, name='')
        var_list = {}
        reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
        for key in reader.get_variable_to_shape_map():
            try:
                tensor = sess.graph.get_tensor_by_name(key + ':0')
            except KeyError:
                continue

            var_list[key] = tensor

        saver = tf.compat.v1.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt_path)
        g = tf.compat.v1.get_default_graph()
        for node_name in sorted_node_names:
            if '_quant/LinearQuant' in node_name:
                quant_scale_node = graph_utils.node_from_map(node_map, node_name)
                quant_source_node = graph_utils.node_from_map(node_map, quant_scale_node.input[0])
                quant_sf_node = graph_utils.node_from_map(node_map, quant_scale_node.input[1])
                quant_bitwidth_node = graph_utils.node_from_map(node_map, node_name.replace('LinearQuant', 'bitwidth'))
                quant_issigned_node = graph_utils.node_from_map(node_map, node_name.replace('LinearQuant', 'is_signed'))
                if node_name.replace('LinearQuant', 'pof2') in node_map:
                    quant_consumer_node = graph_utils.node_from_map(node_map, node_name.replace('LinearQuant', 'pof2'))
                    quant_threshold_node = graph_utils.node_from_map(node_map, quant_consumer_node.input[0].replace('/read', ''))
                sf_tensor = g.get_tensor_by_name(quant_sf_node.name + ':0')
                sf_value = float(sess.run(sf_tensor))
                bitwidth_value = int(graph_utils.values_from_const(quant_bitwidth_node))
                is_signed_value = int(graph_utils.values_from_const(quant_issigned_node))
                if quant_threshold_node:
                    threshold_tensor = g.get_tensor_by_name(quant_threshold_node.name + ':0')
                    threshold_value = 2 ** float(sess.run(threshold_tensor))
                quant_params[quant_source_node.name] = {'source_node':quant_source_node.name, 
                 'bitwidth':bitwidth_value, 
                 'is_signed':is_signed_value, 
                 'scale_factor':sf_value, 
                 'rounding_mode':quant_scale_node.attr['rounding_mode'].s.decode('utf-8'), 
                 'threshold':threshold_value}
            if '_weight_quant/LinearQuant' in node_name or '_bias_quant/LinearQuant' in node_name:
                quant_scale_node = graph_utils.node_from_map(node_map, node_name)
                quant_source_node = graph_utils.node_from_map(node_map, quant_scale_node.input[0])
                quant_sf_node = graph_utils.node_from_map(node_map, quant_scale_node.input[1])
                dequantized_tensor = g.get_tensor_by_name(quant_scale_node.name + ':0')
                sf_tensor = g.get_tensor_by_name(quant_sf_node.name + ':0')
                real_data, scale_factor = sess.run([dequantized_tensor, sf_tensor])
                quantized_data = real_data / scale_factor
                quant_weights.create_dataset(quant_source_node.name, data=quantized_data)

    with open(json_path, 'w') as (json_file):
        json.dump(quant_params, json_file, sort_keys=True, indent=4)
    print(("Saved quantization parameters to '{}'").format(json_path))
    quant_weights.close()
    print(("Saved quantized weights to '{}'").format(weights_path))
    return input_graph_def


def cast_layers_to_double_precision(input_graph_def):
    """
    Post processing step! Always run this after calibrate_quant_layers step and
    before split_conv_nodes_in_depth step.
    
    This function casts the conv/matmul nodes to use fp64 (double precision) to better
    model a long accumulation (e.g. n > 128 accumulations of 16 bt products can exceed
    23 mantissa bits in fp32.
    
        for i from 1 to 1024:
          acc += wt_j_i  *  in_i
                 ------     ----
                 8 bits  +  8 bits   =   16 bits
    
      Due to the continuous accumulation (e.g., ~1024 times) of 16 bit numbers,
      fp32 is insufficient (only 23 mantissa bits), due to bit growth
        n.2^16 < 2^23
             n < 2^7
             n < 128 (and we need ~1024 accumulations)
    """
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for node in input_graph_def.node:
        if node.op in frozenset(['BatchMatMulV2', 'MatMul', 'Conv2D', 'BatchMatMul']):
            _cast_node_to_double_precision(node, nodes_to_skip, new_nodes)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def cast_avgpool_to_double_precision(input_graph_def):
    """
    Post processing step! Always run this after calibrate_quant_layers step and
    before split_conv_nodes_in_depth step.
    
    Cannot be combined with cast_layers_to_double_precision, since graph modifications
    need to be disjoint, otherwise node_map goes stale.
    
    This function casts the avgpool nodes to use fp64 (double precision) to better
    model a long accumulation (e.g. n > 128 accumulations of 16 bt products can exceed
    23 mantissa bits in fp32.
    
        for i from 1 to 1024:
          acc += wt_j_i  *  in_i
                 ------     ----
                 8 bits  +  8 bits   =   16 bits
    
      Due to the continuous accumulation (e.g., ~1024 times) of 16 bit numbers,
      fp32 is insufficient (only 23 mantissa bits), due to bit growth
        n.2^16 < 2^23
             n < 2^7
             n < 128 (and we need ~1024 accumulations)
    """
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    for node in input_graph_def.node:
        if node.op in frozenset(['DepthwiseConv2dNative']):
            if '_from_avgpool' in node.name:
                _cast_node_to_double_precision(node, nodes_to_skip, new_nodes)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def _cast_node_to_double_precision(node, nodes_to_skip, new_nodes):
    layer_node = node
    input_node = node_map[layer_node.input[0]]
    weight_node = node_map[layer_node.input[1]]
    consumer_nodes = output_node_map[layer_node.name]
    for consumer_node_name, input_index in consumer_nodes.items():
        quant_node = node_map[consumer_node_name]
        if quant_node.op != 'LinearQuant':
            raise ValueError('Expected %s to be followed by LinearQuant op, instead got %s op.' % (layer_node.name, quant_node.op))

    quant_in1_node = node_map[quant_node.input[1]]
    quant_in2_node = node_map[quant_node.input[2]]
    quant_in3_node = node_map[quant_node.input[3]]
    layer_node.attr['T'].type = 2
    quant_node.attr['T'].type = 2
    scope, sep, name = layer_node.name.rpartition('/')
    input_cast_node = tf.compat.v1.NodeDef()
    input_cast_node.op = 'Cast'
    input_cast_node.name = scope + sep + 'cast1_' + name
    input_cast_node.input.extend([input_node.name])
    input_cast_node.attr['DstT'].type = 2
    input_cast_node.attr['SrcT'].type = 1
    input_cast_node.attr['Truncate'].b = False
    weight_cast_node = tf.compat.v1.NodeDef()
    weight_cast_node.op = 'Cast'
    weight_cast_node.name = scope + sep + 'cast2_' + name
    weight_cast_node.input.extend([weight_node.name])
    weight_cast_node.attr['DstT'].type = 2
    weight_cast_node.attr['SrcT'].type = 1
    weight_cast_node.attr['Truncate'].b = False
    output_cast_node = tf.compat.v1.NodeDef()
    output_cast_node.op = 'Cast'
    output_cast_node.name = scope + sep + 'cast3_' + name
    output_cast_node.input.extend([quant_node.name])
    output_cast_node.attr['DstT'].type = 1
    output_cast_node.attr['SrcT'].type = 2
    output_cast_node.attr['Truncate'].b = False
    scope, sep, name = quant_node.name.rpartition('/')
    in1_cast_node = tf.compat.v1.NodeDef()
    in1_cast_node.op = 'Cast'
    in1_cast_node.name = scope + sep + 'cast1_' + name
    in1_cast_node.input.extend([quant_in1_node.name])
    in1_cast_node.attr['DstT'].type = 2
    in1_cast_node.attr['SrcT'].type = 1
    in1_cast_node.attr['Truncate'].b = False
    in2_cast_node = tf.compat.v1.NodeDef()
    in2_cast_node.op = 'Cast'
    in2_cast_node.name = scope + sep + 'cast2_' + name
    in2_cast_node.input.extend([quant_in2_node.name])
    in2_cast_node.attr['DstT'].type = 2
    in2_cast_node.attr['SrcT'].type = 1
    in2_cast_node.attr['Truncate'].b = False
    in3_cast_node = tf.compat.v1.NodeDef()
    in3_cast_node.op = 'Cast'
    in3_cast_node.name = scope + sep + 'cast3_' + name
    in3_cast_node.input.extend([quant_in3_node.name])
    in3_cast_node.attr['DstT'].type = 2
    in3_cast_node.attr['SrcT'].type = 1
    in3_cast_node.attr['Truncate'].b = False
    consumer_nodes = output_node_map[quant_node.name]
    for consumer_node_name, input_index in consumer_nodes.items():
        output_node = node_map[consumer_node_name]
        del output_node.input[input_index]
        output_node.input.insert(input_index, output_cast_node.name)

    del layer_node.input[:]
    layer_node.input.extend([input_cast_node.name, weight_cast_node.name])
    del quant_node.input[1:]
    quant_node.input.extend([in1_cast_node.name, in2_cast_node.name, in3_cast_node.name])
    new_nodes.extend([input_cast_node, weight_cast_node, output_cast_node,
     in1_cast_node, in2_cast_node, in3_cast_node])


def split_conv_nodes_in_depth(input_graph_def, conv_split_list, json_path):
    """
    Post processing step! Always run this after calibrate_quant_layers step, because
    this step relies on the precalibrated thresholds.
    
    This function models internal precision of convolution and intermediate accumulation
    by splitting the Conv2D node along the input channel dimension (depth-wise splitting)
    into several Conv2D nodes operating on different sub-volumes as they do on target HW.
    
    conv_split_list is the path to a .txt file containing conv node names 
    to be split, input channel depth, split depth (comma separated). E.g.
    
    conv1_7x7_s2/Conv2D, 3, 4
    conv2_3x3_reduce/Conv2D, 64, 64
    conv2_3x3/Conv2D, 64, 16
    inception_3a_1x1/Conv2D, 192, 192
    inception_3a_pool_proj/Conv2D, 192, 64
    inception_3a_3x3/Conv2D, 96, 16
    inception_3a_5x5/Conv2D, 16, 16
    
    Split is only done when split_depth < channel_depth!
    """
    global node_map
    global output_node_map
    node_map = graph_utils.create_node_map(input_graph_def)
    output_node_map = graph_utils.create_output_node_map(input_graph_def)
    nodes_to_skip = {}
    new_nodes = []
    conv_layers_split_specs = []
    with open(conv_split_list, 'r') as (f):
        for line in f.readlines():
            if line != '\n':
                conv_layers_split_specs.append(line.strip().split(', '))

    with open(json_path, 'r') as (json_file):
        quant_params = json.load(json_file)
    for conv_layer_split_spec in conv_layers_split_specs:
        conv_node_name = conv_layer_split_spec[0]
        channel_depth = int(conv_layer_split_spec[1])
        split_depth = int(conv_layer_split_spec[2])
        if split_depth < channel_depth:
            quotient = channel_depth // split_depth
            remainder = channel_depth % split_depth
            split_sizes = [split_depth for i in range(quotient)]
            if remainder != 0:
                split_sizes.append(channel_depth % split_depth)
            threshold_value = quant_params[conv_node_name]['threshold']
            _split_conv_along_depth(conv_node_name, split_sizes, threshold_value, nodes_to_skip, new_nodes)

    output_graph_def = tf.compat.v1.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = tf.compat.v1.NodeDef()
        new_node.CopyFrom(node)
        output_graph_def.node.extend([new_node])

    output_graph_def.node.extend(new_nodes)
    return output_graph_def


def _split_conv_along_depth(conv_node_name, split_sizes, threshold_value, nodes_to_skip, new_nodes):
    conv_node = graph_utils.node_from_map(node_map, conv_node_name)
    nodes_to_skip[conv_node.name] = True
    scope, sep, name = conv_node.name.rpartition('/')
    strides = list(conv_node.attr['strides'].list.i)
    padding = conv_node.attr['padding'].s
    use_cudnn_on_gpu = conv_node.attr['use_cudnn_on_gpu'].b
    data_format = conv_node.attr['data_format'].s
    dilations = list(conv_node.attr['dilations'].list.i)
    if conv_node.attr['T'].type == 1:
        dtype = tf.float32
    else:
        if conv_node.attr['T'].type == 2:
            dtype = tf.float64
        else:
            raise ValueError('Invalid dtype enum "%d" for node: %s' % (conv_node.attr['T'].type, conv_node.name))
        quant_scope = scope + sep + name + '_' + 'layer_quant'
        quant_bitwidth_node_name = quant_scope + '/bitwidth'
        quant_bitwidth_node = graph_utils.node_from_map(node_map, quant_bitwidth_node_name)
        bitwidth_value = int(graph_utils.values_from_const(quant_bitwidth_node))
        quant_issigned_node_name = quant_scope + '/is_signed'
        quant_issigned_node = graph_utils.node_from_map(node_map, quant_issigned_node_name)
        is_signed_value = int(graph_utils.values_from_const(quant_issigned_node))
        if is_signed_value == 1.0:
            bitwidth_value = -bitwidth_value
        temp_graph = tf.Graph()
        with temp_graph.as_default():
            with temp_graph.name_scope(scope + sep + name + '_dw' + sep):
                input_tensor = tf.compat.v1.placeholder(dtype=dtype)
                weight_tensor = tf.compat.v1.placeholder(dtype=dtype)
                input_split_tensors = tf.split(input_tensor, split_sizes, axis=3, name='input_split')
                weight_split_tensors = tf.split(weight_tensor, split_sizes, axis=2, name='weight_split')
                output_split_tensors = []
                for i, split_size in enumerate(split_sizes):
                    output_tensor = tf.nn.conv2d(input_split_tensors[i], weight_split_tensors[i], strides, padding, use_cudnn_on_gpu, data_format.decode('utf-8'), dilations)
                    if i == 0:
                        with temp_graph.name_scope(scope + sep + name + '_dw' + sep + 'layer_quant_' + str(i) + sep):
                            output_tensor = _linear_quant_v2(output_tensor, is_training=False,
                              bitwidth=bitwidth_value,
                              dtype=dtype,
                              precalibrated_threshold=threshold_value)
                    else:
                        if i == len(split_sizes) - 1:
                            output_tensor = tf.add(output_tensor, output_split_tensors[i - 1])
                        else:
                            output_tensor = tf.add(output_tensor, output_split_tensors[i - 1])
                            with temp_graph.name_scope(scope + sep + name + '_dw' + sep + 'layer_quant_' + str(i) + sep):
                                output_tensor = _linear_quant_v2(output_tensor, is_training=False,
                                  bitwidth=bitwidth_value,
                                  dtype=dtype,
                                  precalibrated_threshold=threshold_value)
                        output_split_tensors.append(output_tensor)

        replace_map = {}
        replace_map[input_tensor.op.node_def.name] = graph_utils.node_name_from_input(conv_node.input[0])
        replace_map[weight_tensor.op.node_def.name] = graph_utils.node_name_from_input(conv_node.input[1])
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

        consumer_nodes = output_node_map[conv_node.name]
        for consumer_node_name, input_index in consumer_nodes.items():
            consumer_node = node_map[consumer_node_name]
            del consumer_node.input[input_index]
            consumer_node.input.insert(input_index, output_split_tensors[-1].op.node_def.name)


def _cdf_measure(x, y, measure_name='Kullback-Leibler-J'):
    """
    Reference:
    https://gitenterprise.xilinx.com/ssettle/TensorXT/blob/master/tensorxt_calibrate_simple.ipynb
    
    Ref paper:
    "Non-parametric Information-Theoretic Measures of One-Dimensional
    Distribution Functions from Continuous Time Series" - Paolo D Alberto et al.
    https://epubs.siam.org/doi/abs/10.1137/1.9781611972795.59
    https://epubs.siam.org/doi/pdf/10.1137/1.9781611972795.59
    
    measure_names_symm = ['Camberra', 'Chi-Squared', 'Cramer-von Mises', 'Euclidean', 
               'Hellinger', 'Jin-L', 'Jensen-Shannon', 'Kolmogorov-Smirnov', 
               'Kullback-Leibler-J', 'Variational']
    measure_names_asym = ['Jin-K', 'Kullback-Leibler-I']
    measure_names_excl = ['Bhattacharyya', 'Phi', 'Xi']
    """
    if measure_name == 'Bhattacharyya':
        return np.sum(np.sqrt(x * y))
    elif measure_name == 'Camberra':
        return np.sum(np.abs(x - y) / (x + y))
    elif measure_name == 'Chi-Squared':
        return np.sum(np.power(x - y, 2.0) / x)
    elif measure_name == 'Cramer-von Mises':
        return np.sum(np.power(x - y, 2.0))
    elif measure_name == 'Euclidean':
        return np.power(np.sum(np.power(x - y, 2.0)), 0.5)
    elif measure_name == 'Hellinger':
        return np.power(np.sum(np.sqrt(x) - np.sqrt(y)), 2.0) / 2.0
    elif measure_name == 'Jin-K':
        return _cdf_measure(x, (x + y) / 2.0, 'Kullback-Leibler-I')
    elif measure_name == 'Jin-L':
        return _cdf_measure(x, (x + y) / 2.0, 'Kullback-Leibler-I') + _cdf_measure(y, (x + y) / 2.0, 'Kullback-Leibler-I')
    elif measure_name == 'Jensen-Shannon':
        return (_cdf_measure(x, (x + y) / 2.0, 'Kullback-Leibler-I') + _cdf_measure(y, (x + y) / 2.0, 'Kullback-Leibler-I')) / 2.0
    elif measure_name == 'Kolmogorov-Smirnov':
        return np.max(np.abs(x - y))
    elif measure_name == 'Kullback-Leibler-I':
        return np.sum(x * np.log2(x / y))
    elif measure_name == 'Kullback-Leibler-J':
        return np.sum((x - y) * np.log2(x / y))
    elif measure_name == 'Phi':
        return np.max(np.abs(x - y) / np.sqrt(np.minimum((x + y) / 2.0, 1 - (x + y) / 2.0)))
    elif measure_name == 'Variational':
        return np.sum(np.abs(x - y))
    elif measure_name == 'Xi':
        return np.max(np.abs(x - y) / np.sqrt((x + y) / 2.0 * (1 - (x + y) / 2.0)))
    else:
        return _cdf_measure(x, y, 'Kullback-Leibler-J')


def _compute_threshold(data, bitwidth, is_signed, bins='sqrt'):
    """
    Reference:
    https://gitenterprise.xilinx.com/ssettle/ristretto/blob/master/python/quantize_dynamic_fixed_point.py
    
    Ref paper (Algorithm 1):
    "Quantizing Convolutional Neural Networks for Low-Power
    High-Throughput Inference Engines" - Sean Settle et al.
    https://arxiv.org/abs/1805.07941
    https://arxiv.org/pdf/1805.07941.pdf
    """
    mn = 0
    mx = np.max(np.abs(data))
    hist, bin_edges = np.histogram(np.abs(data), bins, range=(mn, mx), density=True)
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    n = pow(2, bitwidth - is_signed)
    threshold = []
    d = []
    if n + 1 > len(bin_edges) - 1:
        threshold_final = bin_edges[-1]
        return threshold_final
    else:
        for i in range(n + 1, len(bin_edges), 1):
            threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
            threshold = np.concatenate((threshold, [threshold_tmp]))
            p = np.copy(cdf)
            p[i - 1:] = 1
            x = np.linspace(0.0, 1.0, n)
            xp = np.linspace(0.0, 1.0, i)
            fp = p[:i]
            p_interp = np.interp(x, xp, fp)
            x = np.linspace(0.0, 1.0, i)
            xp = np.linspace(0.0, 1.0, n)
            fp = p_interp
            q_interp = np.interp(x, xp, fp)
            q = np.copy(p)
            q[:i] = q_interp
            d_tmp = _cdf_measure(cdf[np.nonzero(cdf)], q[np.nonzero(cdf)], 'Kullback-Leibler-J')
            d = np.concatenate((d, [d_tmp]))

        threshold_final = threshold[np.argmin(d)]
        return threshold_final