#! /usr/bin/env python 3.6 (3379)
#coding=utf-8
# Compiled at: 2019-10-23 20:32:36
#Powered by BugScaner
#http://tools.bugscaner.com/
#如果觉得不错,请分享给你朋友使用吧!
"""
Graffitist: Graph Transforms to Quantize and Retrain Deep Neural Nets in TensorFlow.

@ author: Sambhav Jain
"""
import argparse, os, re
from collections import OrderedDict
from google.protobuf import text_format as pbtf
import tensorflow as tf
from graffitist import transforms
from graffitist.utils import graph_utils
available_transforms = sorted((name for name in transforms.__dict__ if name.islower() if not name.startswith('__') if callable(transforms.__dict__[name])))
parser = argparse.ArgumentParser(description='Graffitist: Graph Transforms to Quantize and Retrain Deep Neural Nets in TensorFlow')
parser.add_argument('--in_graph', metavar='G', type=str, required=True, help='path to input graph (.meta/.pb/frozen.pb)')
parser.add_argument('--out_graph', metavar='G', type=str, required=True, help='path to output graph (.pb)')
parser.add_argument('--inputs', metavar='N', nargs='+', type=str, required=True, help='input node names')
parser.add_argument('--outputs', metavar='N', nargs='+', type=str, required=True, help='output node names')
parser.add_argument('--input_shape', metavar='H,W,C', type=str, required=True, help='input shape excluding batch size (H, W, C)')
parser.add_argument('--saved_model_tag', metavar='tag', type=str, required=False, help='SavedModel tag to identify the MetaGraphDef to load')
parser.add_argument('--transforms', metavar='T', nargs='+', type=str, help='list of transforms to apply: ' + (' | ').join(available_transforms))
parser.add_argument('--superuser', dest='superuser', action='store_true', help=argparse.SUPPRESS)

def parse_transforms(transform_list):
    """
    Reads the transforms list and returns an ordered dict mapping 
    transform names to another ordered dict of the corresponding *args.
    
    Expects transform to be in the format "name(arg1=arg1, arg2=arg2, arg3=[arg3a,arg3b])"
    
    Returns OrderedDict([('name', OrderedDict([('arg1', 'arg1'), ('arg2', 'arg2'), ('arg3', '[arg3a,arg3b]')]))])
    """
    transform_dict = OrderedDict()
    if isinstance(transform_list, str):
        transform_list = [transform_list]
    for t in transform_list:
        name_args = re.split('[()]', t)
        if len(name_args) == 1:
            transform_dict[name_args[0]] = None
        else:
            arg_dict = OrderedDict()
            arg_list = re.split(', ', name_args[1])
            for arg in arg_list:
                key_val = re.split('=', arg)
                arg_dict[key_val[0]] = key_val[1]

            transform_dict[name_args[0]] = arg_dict

    return transform_dict


def check_op_compatibility(input_graph_def):
    """
    The list here is obtained from
    https://www.tensorflow.org/api_docs/cc/group/training-ops
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/quantize/python/quantize_graph.py#L240-L279
    """
    training_ops = frozenset([
     'ApplyAdagrad', 'ApplyAdagradDA', 'ApplyAdam', 'ApplyAddSign',
     'ApplyCenteredRMSProp', 'ApplyFtrl', 'ApplyFtrlV2',
     'ApplyGradientDescent', 'ApplyMomentum', 'ApplyPowerSign',
     'ApplyProximalAdagrad', 'ApplyProximalGradientDescent', 'ApplyRMSProp',
     'ResourceApplyAdadelta', 'ResourceApplyAdagrad', 'ResourceApplyAdagradDA',
     'ResourceApplyAdam', 'ResourceApplyAddSign',
     'ResourceApplyCenteredRMSProp', 'ResourceApplyFtrl',
     'ResourceApplyFtrlV2', 'ResourceApplyGradientDescent',
     'ResourceApplyMomentum', 'ResourceApplyPowerSign',
     'ResourceApplyProximalAdagrad', 'ResourceApplyProximalGradientDescent',
     'ResourceApplyRMSProp', 'ResourceSparseApplyAdadelta',
     'ResourceSparseApplyAdagrad', 'ResourceSparseApplyAdagradDA',
     'ResourceSparseApplyCenteredRMSProp', 'ResourceSparseApplyFtrl',
     'ResourceSparseApplyFtrlV2', 'ResourceSparseApplyMomentum',
     'ResourceSparseApplyProximalAdagrad',
     'ResourceSparseApplyProximalGradientDescent',
     'ResourceSparseApplyRMSProp', 'SparseApplyAdadelta', 'SparseApplyAdagrad',
     'SparseApplyAdagradDA', 'SparseApplyCenteredRMSProp', 'SparseApplyFtrl',
     'SparseApplyFtrlV2', 'SparseApplyMomentum', 'SparseApplyProximalAdagrad',
     'SparseApplyProximalGradientDescent', 'SparseApplyRMSProp'])
    supported_ops = frozenset([
     'Mul', 'PlaceholderWithDefault', 'Const', 'Assign', 'Identity', 'Mean',
     'VariableV2', 'BiasAdd', 'Conv2D', 'Sub', 'MaxPool', 'Relu', 'LeakyRelu',
     'MatMul', 'Softmax', 'Reshape', 'Add', 'Placeholder', 'RandomUniform',
     'AvgPool', 'ConcatV2', 'L2Loss', 'Squeeze', 'Fill', 'Shape', 'Maximum',
     'FusedBatchNorm', 'TruncatedNormal', 'DepthwiseConv2dNative', 'Pack',
     'StridedSlice', 'Pad', 'Relu6', 'AssignSub', 'NoOp', 'RestoreV2', 'SaveV2',
     'AssignVariableOp', 'ReadVariableOp', 'VarHandleOp', 'VarIsInitializedOp',
     'Transpose'])
    op_types = set([node.op for node in input_graph_def.node])
    train_op_list = op_types.intersection(training_ops)
    if train_op_list:
        raise ValueError('Training op(s) found in input graph: %s; exiting' % train_op_list)
    unsupported_op_list = op_types - op_types.intersection(supported_ops)
    if unsupported_op_list:
        raise ValueError('Unsupported op(s) found in input graph: %s; exiting' % unsupported_op_list)


def main():
    global args
    args = parser.parse_args()
    if not tf.io.gfile.exists(args.in_graph):
        raise ValueError(("Input graph file '{}' does not exist!").format(args.in_graph))
    tf_collections = {}
    if args.in_graph.endswith('.meta'):
        saver = tf.compat.v1.train.import_meta_graph(args.in_graph, clear_devices=True)
        g = tf.compat.v1.get_default_graph()
        for key in g.get_all_collection_keys():
            variables = [node.name for node in g.get_collection(key) if isinstance(node, tf.Variable)]
            operations = [node.name for node in g.get_collection(key) if isinstance(node, tf.Operation)]
            tensors = [node.name for node in g.get_collection(key) if isinstance(node, tf.Tensor)]
            tf_collections[key] = {'variables':variables,  'operations':operations,  'tensors':tensors}

        input_graph_def = g.as_graph_def()
    else:
        if args.in_graph.endswith('saved_model.pb'):
            if not args.saved_model_tag:
                raise ValueError("To load a SavedModel, please provide the flag '--saved_model_tag <tag>'.")
            with tf.Session(graph=tf.Graph()) as (sess):
                tf.saved_model.loader.load(sess, [args.saved_model_tag], args.in_graph.rsplit('/', 1)[0])
                g = tf.compat.v1.get_default_graph()
                for key in g.get_all_collection_keys():
                    variables = [node.name for node in g.get_collection(key) if isinstance(node, tf.Variable)]
                    operations = [node.name for node in g.get_collection(key) if isinstance(node, tf.Operation)]
                    tensors = [node.name for node in g.get_collection(key) if isinstance(node, tf.Tensor)]
                    tf_collections[key] = {'variables':variables,  'operations':operations,  'tensors':tensors}

                input_graph_def = g.as_graph_def()
        else:
            input_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(args.in_graph, 'rb') as (f):
                if args.in_graph.endswith('.pbtxt'):
                    pbtf.Parse(f.read(), input_graph_def)
                else:
                    input_graph_def.ParseFromString(f.read())
        graph_utils.ensure_graph_is_valid(input_graph_def)
        print(("Input graph loaded from '{}'").format(args.in_graph))
        print(('Input graph node count = {}').format(len(input_graph_def.node)))
        if not args.superuser:
            check_op_compatibility(input_graph_def)
        output_graph_def = input_graph_def
        ckpt_dir = os.path.dirname(args.in_graph)
        transform_is_applied = {}
        for t in available_transforms:
            transform_is_applied[t] = False

        if args.transforms:
            transform_dict = parse_transforms(args.transforms)
            for t in transform_dict:
                if t == 'freeze_graph':
                    output_graph_def = transforms.freeze_graph(output_graph_def, ckpt_dir=ckpt_dir,
                      output_node_names=args.outputs)
                else:
                    if t == 'fold_batch_norms_inplace':
                        if not transform_is_applied['freeze_graph']:
                            raise ValueError("Transform 'fold_batch_norms_inplace' requires 'freeze_graph' to be applied first!")
                        output_graph_def = transforms.fold_batch_norms_inplace(output_graph_def)
                    else:
                        if t == 'fix_input_shape':
                            output_graph_def = transforms.fix_input_shape(output_graph_def, input_node_names=args.inputs,
                              input_shape=args.input_shape)
                        else:
                            if t == 'fold_batch_norms':
                                if not transform_is_applied['fix_input_shape']:
                                    raise ValueError("Transform 'fold_batch_norms' requires 'fix_input_shape' to be applied first!")
                                if transform_is_applied['remove_training_nodes']:
                                    raise ValueError("Transform 'fold_batch_norms' cannot be applied after 'remove_training_nodes'!")
                                if transform_dict[t]:
                                    output_graph_def = transforms.fold_batch_norms(output_graph_def, **transform_dict[t])
                                else:
                                    output_graph_def = transforms.fold_batch_norms(output_graph_def)
                                output_graph_def = graph_utils.add_static_shapes(output_graph_def)
                            else:
                                if t == 'remove_training_nodes':
                                    if transform_is_applied['preprocess_layers']:
                                        raise ValueError("Transform 'remove_training_nodes' cannot be applied after 'preprocess_layers'")
                                    if isinstance(args.outputs, list):
                                        protected_nodes = args.outputs
                                    else:
                                        protected_nodes = list(args.outputs)
                                    if transform_dict[t]:
                                        output_graph_def = transforms.remove_training_nodes(output_graph_def, 
                                         protected_nodes, **transform_dict[t])
                                    else:
                                        output_graph_def = transforms.remove_training_nodes(output_graph_def, protected_nodes)
                                    output_graph_def = graph_utils.add_static_shapes(output_graph_def)
                                else:
                                    if t == 'strip_unused_nodes':
                                        if transform_dict[t]:
                                            output_graph_def = transforms.strip_unused_nodes(input_node_names=args.inputs, 
                                             output_node_names=args.outputs, **transform_dict[t])
                                        else:
                                            output_graph_def = transforms.strip_unused_nodes(output_graph_def, input_node_names=args.inputs,
                                              output_node_names=args.outputs)
                                        output_graph_def = graph_utils.add_static_shapes(output_graph_def)
                                    else:
                                        if t == 'preprocess_layers':
                                            if not transform_is_applied['fix_input_shape']:
                                                raise ValueError("Transform 'preprocess_layers' requires 'fix_input_shape' to be applied first!")
                                            output_graph_def = transforms.preprocess_layers(output_graph_def)
                                            output_graph_def = graph_utils.add_static_shapes(output_graph_def)
                                        else:
                                            if t == 'quantize':
                                                if transform_is_applied['freeze_graph']:
                                                    raise ValueError("Transform 'quantize' cannot be applied after 'freeze_graph'!")
                                                if not transform_is_applied['fix_input_shape']:
                                                    raise ValueError("Transform 'quantize' requires 'fix_input_shape' to be applied first!")
                                                if not transform_is_applied['fold_batch_norms']:
                                                    raise ValueError("Transform 'quantize' requires 'fold_batch_norms' to be applied first!")
                                                if not transform_is_applied['preprocess_layers']:
                                                    raise ValueError("Transform 'quantize' requires 'preprocess_layers' to be applied first!")
                                                calib_path = os.path.join(ckpt_dir, 'calibration_set.npy')
                                                json_path = os.path.join(ckpt_dir, 'quantization_params.json')
                                                weights_path = os.path.join(ckpt_dir, 'quantized_weights.h5')
                                                if args.in_graph.endswith('saved_model.pb'):
                                                    ckpt_dir = os.path.join(ckpt_dir, 'variables')
                                                if transform_dict[t]:
                                                    output_graph_def = transforms.quantize(input_node_names=args.inputs, 
                                                     ckpt_dir=ckpt_dir, 
                                                     calib_path=calib_path, 
                                                     json_path=json_path, 
                                                     weights_path=weights_path, 
                                                     tf_collections=tf_collections, 
                                                     verbose=args.superuser, **transform_dict[t])
                                                else:
                                                    output_graph_def = transforms.quantize(output_graph_def, input_node_names=args.inputs,
                                                      ckpt_dir=ckpt_dir,
                                                      calib_path=calib_path,
                                                      json_path=json_path,
                                                      weights_path=weights_path,
                                                      tf_collections=tf_collections,
                                                      verbose=args.superuser)
                                                output_graph_def = graph_utils.add_static_shapes(output_graph_def)
                                            else:
                                                print(("Transform '{}' is invalid.").format(t))
                                                continue
                                            transform_is_applied[t] = True
                                            print(("Output graph node count = {} after transform '{}'").format(len(output_graph_def.node), t))

        graph_utils.ensure_graph_is_valid(output_graph_def)
        with tf.io.gfile.GFile(args.out_graph, 'wb') as (f):
            f.write(output_graph_def.SerializeToString())
        print(('Output graph node count = {}').format(len(output_graph_def.node)))
        print(("Output graph saved to '{}'").format(args.out_graph))


if __name__ == '__main__':
    main()