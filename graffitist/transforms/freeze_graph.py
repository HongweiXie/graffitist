#! /usr/bin/env python 3.6 (3379)
#coding=utf-8
# Compiled at: 2019-10-18 19:07:52
#Powered by BugScaner
#http://tools.bugscaner.com/
#如果觉得不错,请分享给你朋友使用吧!
"""
Converts checkpoint variables into Const ops in a standalone GraphDef file.

@ author: Sambhav Jain
"""
__all__ = [
 'freeze_graph']
import tensorflow as tf

def freeze_graph(input_graph_def, ckpt_dir, output_node_names):
    """
    This script is designed to take a GraphDef proto, and a set of variable values
    stored in a checkpoint file, and output a GraphDef with all of the variable ops
    converted into const ops containing the values of the variables.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/tools/freeze_graph.py
    """
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    ckpt_path = ckpt.model_checkpoint_path
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
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names)
    return output_graph_def