#! /usr/bin/env python 3.6 (3379)
#coding=utf-8
# Compiled at: 2019-10-18 19:07:52
#Powered by BugScaner
#http://tools.bugscaner.com/
#如果觉得不错,请分享给你朋友使用吧!
"""
Quantize graph

@ author: Sambhav Jain
"""
__all__ = [
 'quantize']
import tensorflow as tf
from graffitist.utils import quantize_utils

def quantize(input_graph_def, input_node_names, ckpt_dir, calib_path, json_path, weights_path, tf_collections, verbose, weight_bits=-8, activation_bits=-8, layer_bits=-16, relu_bits=8, avgpool_reciprocal_bits=8, avgpool_bits=8, first_layer=None, last_layer=None, layer_merge_list=None, conv_split_list=None, calibrate_quant_layers=True, dump_quant_params=False, bit_accurate=False, is_training=False):
    weight_bits = int(weight_bits)
    activation_bits = int(activation_bits)
    layer_bits = int(layer_bits)
    relu_bits = int(relu_bits)
    avgpool_bits = int(avgpool_bits)
    avgpool_reciprocal_bits = int(avgpool_reciprocal_bits)
    calibrate_quant_layers = calibrate_quant_layers == 'True' or calibrate_quant_layers == True
    dump_quant_params = dump_quant_params == 'True' or dump_quant_params == True
    bit_accurate = bit_accurate == 'True' or bit_accurate == True
    is_training = is_training == 'True' or is_training == True
    output_graph_def = input_graph_def
    output_graph_def = quantize_utils.quantize_layers(output_graph_def,
      weight_bits=weight_bits,
      activation_bits=activation_bits,
      layer_bits=layer_bits,
      relu_bits=relu_bits,
      first_layer=first_layer,
      last_layer=last_layer,
      is_training=is_training)
    output_graph_def = quantize_utils.quantize_input(output_graph_def,
      input_node_names,
      activation_bits=activation_bits,
      is_training=is_training)
    output_graph_def = quantize_utils.quantize_separable_conv(output_graph_def,
      weight_bits=weight_bits,
      activation_bits=activation_bits,
      first_layer=first_layer,
      last_layer=last_layer,
      is_training=is_training)
    output_graph_def = quantize_utils.quantize_rescale(output_graph_def,
      activation_bits=activation_bits,
      relu_bits=relu_bits,
      is_training=is_training)
    output_graph_def = quantize_utils.quantize_avgpool(output_graph_def,
      avgpool_bits=avgpool_bits,
      avgpool_reciprocal_bits=avgpool_reciprocal_bits,
      is_training=is_training)
    output_graph_def = quantize_utils.auto_merge_quant_layers(output_graph_def)
    if layer_merge_list:
        output_graph_def = quantize_utils.manual_merge_quant_layers(output_graph_def,
          layer_merge_list=layer_merge_list)
    if calibrate_quant_layers:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        try:
            ckpt_path = ckpt.model_checkpoint_path
            print(("Using ckpt '{}' for transform 'calibrate_quant_layers'").format(ckpt_path))
        except:
            ckpt_path = ckpt_dir

        output_graph_def = quantize_utils.calibrate_quant_layers(output_graph_def,
          input_node_names,
          ckpt_path=ckpt_path,
          calib_path=calib_path,
          tf_collections=tf_collections,
          is_training=is_training,
          verbose=verbose)
    if dump_quant_params:
        if not is_training:
            if not calibrate_quant_layers:
                print("WARNING: 'calibrate_quant_layers=False' hence INVALID quantization parameters and quantized weights UNLESS ckpt has pre-calibrated/retrained thresholds!")
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            ckpt_path = ckpt.model_checkpoint_path
            print(("Using ckpt '{}' for transform 'dump_quant_params'").format(ckpt_path))
            output_graph_def = quantize_utils.dump_quant_params(output_graph_def,
              ckpt_path=ckpt_path,
              json_path=json_path,
              weights_path=weights_path)
    if bit_accurate:
        if not is_training:
            output_graph_def = quantize_utils.cast_layers_to_double_precision(output_graph_def)
    if bit_accurate:
        if not is_training:
            output_graph_def = quantize_utils.cast_avgpool_to_double_precision(output_graph_def)
    if conv_split_list:
        if not is_training:
            output_graph_def = quantize_utils.split_conv_nodes_in_depth(output_graph_def,
              conv_split_list=conv_split_list,
              json_path=json_path)
    return output_graph_def