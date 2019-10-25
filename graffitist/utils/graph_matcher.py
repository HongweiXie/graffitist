#! /usr/bin/env python 3.6 (3379)
#coding=utf-8
# Compiled at: 2019-10-18 19:07:52
#Powered by BugScaner
#http://tools.bugscaner.com/
#如果觉得不错,请分享给你朋友使用吧!
"""
Utilities that match patterns in a tf.GraphDef.

Reference:
https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/quantize/python/graph_matcher.py

@ author: Sambhav Jain
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc, itertools
from graffitist.utils import graph_utils
_LAYER_TYPES = {
 'Conv2D', 'MatMul', 'BatchMatMul', 'BatchMatMulV2', 'DepthwiseConv2dNative'}
_RELU_TYPES = {
 'Relu', 'Relu6'}

class Pattern(object):
    """The parent class of all patterns (e.g. OpTypePattern and OneofPattern)."""

    @abc.abstractmethod
    def match(self, op):
        """Returns the result of matching op against this pattern."""
        raise NotImplementedError('Method "match" not implemented.')


class OpTypePattern(Pattern):
    """A tree pattern that matches TF expressions with certain op types."""

    def __init__(self, op_type, name=None, inputs=None, ordered_inputs=True):
        """Initializes an OpTypePattern.
        
        Args:
          op_type: string that specifies the allowed types of the root. It can be
            (1) an op type, e.g. 'Conv2D',
            (2) '*', i.e. wildcard, or
            (3) multiple op types separated by '|', e.g., 'Relu|Relu6'.
            We could use regex strings, which might be worthwhile when we have many
            similar TF op types.
          name: Optional string. The name of the pattern that can be looked up in
            MatchResult.
          inputs: Optional list of `Pattern`s or strings that specify the
            patterns for the inputs of a matching op. If None, this pattern accepts
            any inputs of a matching op.
          ordered_inputs: Defaults to True. If False, will match any op that
            matches a permutation of the inputs.
        
        Raises:
          ValueError: if too many inputs are provided when order_inputs is False.
        """
        self._op_type = op_type
        self._name = name
        if inputs is None:
            inputs = []
        if len(inputs) > 8:
            raise ValueError('Only < 8 inputs are allowed when ordered_inputs is False.')
        self._inputs = [input_pattern if isinstance(input_pattern, Pattern) else OpTypePattern(input_pattern) for input_pattern in inputs]
        self._ordered_inputs = ordered_inputs

    @property
    def name(self):
        return self._name

    def match(self, op):
        if self._op_type != '*':
            if op.op not in self._op_type.split('|'):
                return
            match_result = MatchResult()
            match_result.add(self, op)
            if not self._inputs:
                return match_result
            if len(op.input) != len(self._inputs):
                return
            input_patterns_list = [self._inputs]
            if not self._ordered_inputs:
                input_patterns_list = list(itertools.permutations(self._inputs))
            for input_patterns in input_patterns_list:
                match_failed = False
                for input_node_name, input_pattern in zip(op.input, input_patterns):
                    input_op = graph_utils.node_from_map(node_map, input_node_name)
                    input_match_result = input_pattern.match(input_op)
                    if input_match_result is None:
                        match_failed = True
                        break
                    match_result.merge_from(input_match_result)

                if not match_failed:
                    return match_result


class OneofPattern(Pattern):
    """Matches one of the given sub-patterns."""

    def __init__(self, sub_patterns):
        self._sub_patterns = sub_patterns

    def match(self, op):
        for sub_pattern in self._sub_patterns:
            match_result = sub_pattern.match(op)
            if match_result is not None:
                return match_result


class MatchResult(object):
    r"""Encapsulates the result of a match done by GraphMatcher.
    
    MatchResult contains a map from Pattern to the matching op (tf.NodeDef).
    
    E.g., when we match graph
    
        -         +
       / \y0   y1/ \
      x    split    z
            |
            y         (nodes are ops; edges are going up)
    
    against add_pattern defined as
    
      y1_pattern = OpTypePattern('*')
      z_pattern = OpTypePattern('*')
      add_pattern = OpTypePattern('+', inputs=[y1_pattern, z_pattern])
    
    the matching op of `y1_pattern` is `split`.
    """

    def __init__(self):
        self._pattern_to_op = {}
        self._name_to_pattern = {}

    def add(self, pattern, op):
        self._pattern_to_op[pattern] = op
        if pattern.name is not None:
            if pattern.name in self._name_to_pattern:
                raise ValueError('Name %s is already bound to another pattern' % pattern.name)
            self._name_to_pattern[pattern.name] = pattern

    def _to_pattern(self, pattern_or_name):
        if isinstance(pattern_or_name, Pattern):
            return pattern_or_name
        if isinstance(pattern_or_name, str):
            if pattern_or_name not in self._name_to_pattern:
                return
            return self._name_to_pattern[pattern_or_name]
        raise ValueError('pattern_or_name has type %s. Expect Pattern or str.' % type(pattern_or_name))

    def _get_op(self, pattern_or_name):
        pattern = self._to_pattern(pattern_or_name)
        if pattern is None:
            return
        elif pattern not in self._pattern_to_op:
            return
        else:
            return self._pattern_to_op[pattern]

    def get_op(self, pattern_or_name):
        op = self._get_op(pattern_or_name)
        if op:
            return op

    def merge_from(self, other_match_result):
        self._pattern_to_op.update(other_match_result._pattern_to_op)
        self._name_to_pattern.update(other_match_result._name_to_pattern)


class GraphMatcher(object):
    """Checks if a particular subgraph matches a given pattern."""

    def __init__(self, pattern):
        """Initializes a GraphMatcher.
        
        Args:
          pattern: The `Pattern` against which `GraphMatcher` matches
            subgraphs.
        """
        self._pattern = pattern

    def _match_pattern(self, pattern, op):
        """Returns whether an TF expression rooted at `op` matches `pattern`.
        
        If there is a match, adds to `self._match_result` the matching op with key `pattern`.
        
        Args:
          pattern: An `Pattern`.
          op: A `tf.Operation` to match against the pattern.
        
        Returns:
          True if an TF expression rooted at `op` matches `pattern`.
        """
        match_result = pattern.match(op)
        if match_result is None:
            return False
        else:
            self._match_result.merge_from(match_result)
            return True

    def match_op(self, op):
        """Matches `op` against `self._pattern`.
        
        Args:
          op: `tf.NodeDef` to match against the pattern.
        
        Returns:
          Returns a `MatchResult` if `op` matches the pattern; otherwise, returns
          None.
        """
        self._match_result = MatchResult()
        if not self._match_pattern(self._pattern, op):
            return
        else:
            return self._match_result

    def match_ops(self, ops):
        """Matches each operation in `ops` against `self._pattern`.
        
        Args:
          ops: collection of `tf.NodeDef`s to match against the pattern.
        
        Yields:
          `MatchResult` for each set of `tf.NodeDef`s that matches the pattern.
        """
        for op in ops:
            match_result = self.match_op(op)
            if match_result:
                yield match_result

    def match_graph_def(self, graph_def):
        """Matches each operation in `graph_def` against `self._pattern`.
        
        Args:
          graph_def: `tf.GraphDef` containing operations to match.
        
        Yields:
          `MatchResult` for each set of `tf.NodeDef`s in `graph_def` that matches the pattern.
        """
        global node_map
        node_map = graph_utils.create_node_map(graph_def)
        for match_result in self.match_ops(graph_def.node):
            yield match_result


def find_fused_batch_norms(graph_def):
    """Matches fused batch norm layers in graph to fold.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/quantize/python/fold_batch_norms.py#L142
    
    The following patterns get matched. Nodes surrounded by [] will be
    optionally matched:
    
           input  weight
              |  /
           conv|fc
              |
           [Reshape]   gamma  beta  mean  variance
              |        /      /     /     / 
             FusedBatchNorm                     ----> Graph Matcher 2
                   |
               [Reshape]                        ----> Graph Matcher 1
    
    Args:
      graph_def: tf.GraphDef to perform match on.
    
    Returns:
      List of MatchResult.
    """
    input_pattern = OpTypePattern('*', name='input_pattern')
    weight_pattern = OpTypePattern('*', name='weight_pattern')
    gamma_pattern = OpTypePattern('*', name='gamma_pattern')
    beta_pattern = OpTypePattern('*', name='beta_pattern')
    mean_pattern = OpTypePattern('*', name='mean_pattern')
    variance_pattern = OpTypePattern('*', name='variance_pattern')
    layer_pattern = OpTypePattern(('|').join(_LAYER_TYPES),
      name='layer_pattern',
      inputs=[
     input_pattern, weight_pattern])
    matmul_reshape_pattern = OpTypePattern('Reshape',
      name='matmul_reshape_pattern',
      inputs=[
     layer_pattern, '*'])
    batch_norm_pattern = OpTypePattern('FusedBatchNorm',
      name='batch_norm_pattern',
      inputs=[
     OneofPattern([matmul_reshape_pattern, layer_pattern]),
     gamma_pattern, beta_pattern, mean_pattern, variance_pattern])
    matmul_bn_output_reshape_pattern = OpTypePattern('Reshape',
      name='matmul_bn_output_reshape_pattern',
      inputs=[
     batch_norm_pattern, '*'])
    layer_matches = []
    matched_layer_set = set()
    matmul_bn_reshape_matcher = GraphMatcher(matmul_bn_output_reshape_pattern)
    for match_result in matmul_bn_reshape_matcher.match_graph_def(graph_def):
        layer_node = match_result.get_op('layer_pattern')
        if layer_node.name not in matched_layer_set:
            matched_layer_set.add(layer_node.name)
            layer_matches.append(match_result)

    bn_matcher = GraphMatcher(batch_norm_pattern)
    for match_result in bn_matcher.match_graph_def(graph_def):
        layer_node = match_result.get_op('layer_pattern')
        if layer_node.name not in matched_layer_set:
            matched_layer_set.add(layer_node.name)
            layer_matches.append(match_result)

    return layer_matches


def find_layers_to_quantize(graph_def):
    """Matches layers in graph to quantize.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/quantize/python/quantize.py#L193
    
    The following patterns get matched. Nodes surrounded by [] will be
    optionally matched:
    
    Regular ReLU:
    
           input  weight|folded_weight
              |  /
           conv|fc
              |       bias
     [bn_correction]  / 
              |      /
             biasadd              ----> Graph Matcher 4 (for last layers and projection shortcut in ResNets)
              |
           [bypass]               ----> Graph Matcher 3 (for eltwise add layers without activation afterward - e.g. Mobilenet-v2)
              |
         relu|relu6               ----> Graph Matcher 1 (for relu/relu6)
    
    
    Leaky ReLU:
    
           input  weight|folded_weight
              |  /
           conv|fc
              |       bias
     [bn_correction]  / 
              |      /
             biasadd              ----> Graph Matcher 4 (for last layers and projection shortcut in ResNets)
              |
          leaky-relu              ----> Graph Matcher 2 (for leaky-relu)
    
    Known limitation: Doesn't support bypass followed by leaky-relu
    
    Match replacements:
      If weight|folded_weight is found, weight_quant is added afterwards.
      If conv|fc is found (without bn_correction), layer_quant is added afterwards.
      If conv|fc is found (with bn_correction), layer_quant is added after correction (to ensure auto_merge with bias_quant)
      If bias is found, bias_quant is added afterwards.
      If biasadd is found to be the last layer in the pattern, biasadd_quant is added afterwards.
      If bypass is found, biasadd_quant is added before (not after, as eltwadd_quant or act_quant will be added after anyway).
      If bypass is found without activation, eltwadd_quant will be added afterwards.
      If activation is found, act_quant is added afterwards.
      If activation is leaky relu, alpha*x is also quantized appropriately.
    
    Args:
      graph_def: tf.GraphDef to perform match on.
    
    Returns:
      List of MatchResult.
    """
    input_pattern = OpTypePattern('*', name='input_pattern')
    weight_var_pattern = OpTypePattern('Variable|VariableV2', name='weight_var_pattern')
    weight_identity_pattern = OpTypePattern('Identity',
      name='weight_identity_pattern',
      inputs=[
     weight_var_pattern])
    weight_cast_pattern = OpTypePattern('Cast',
      name='weight_cast_pattern',
      inputs=[
     OneofPattern([
      weight_identity_pattern, weight_var_pattern])])
    weight_resource_var_pattern = OpTypePattern('ReadVariableOp', name='weight_resource_var_pattern')
    frozen_weight_pattern = OpTypePattern('Const', name='frozen_weight_pattern')
    folded_weight_pattern = OpTypePattern('Mul', name='folded_weight_pattern')
    layer_pattern = OpTypePattern(('|').join(_LAYER_TYPES),
      name='layer_pattern',
      inputs=[
     input_pattern,
     OneofPattern([weight_var_pattern,
      weight_identity_pattern,
      weight_cast_pattern,
      weight_resource_var_pattern,
      frozen_weight_pattern,
      folded_weight_pattern])],
      ordered_inputs=False)
    bn_correction_pattern = OpTypePattern('Mul',
      name='bn_correction_pattern',
      inputs=[
     '*', layer_pattern],
      ordered_inputs=False)
    bias_pattern = OpTypePattern('*', name='bias_pattern')
    bias_add_pattern = OpTypePattern('Add|BiasAdd',
      name='bias_add_pattern',
      inputs=[
     OneofPattern([
      layer_pattern, bn_correction_pattern]),
     bias_pattern],
      ordered_inputs=False)
    bypass_pattern = OpTypePattern('Add',
      name='bypass_pattern',
      inputs=[
     '*', bias_add_pattern],
      ordered_inputs=False)
    activation_relu_pattern = OpTypePattern(('|').join(_RELU_TYPES),
      name='activation_relu_pattern',
      inputs=[
     OneofPattern([
      bias_add_pattern, bypass_pattern])])
    leaky_relu_alpha_pattern = OpTypePattern('*', name='leaky_relu_alpha_pattern')
    leaky_relu_alpha_x_pattern = OpTypePattern('Mul',
      name='leaky_relu_alpha_x_pattern',
      inputs=[
     leaky_relu_alpha_pattern, bias_add_pattern],
      ordered_inputs=False)
    activation_leakyrelu_pattern = OpTypePattern('Maximum',
      name='activation_leakyrelu_pattern',
      inputs=[
     leaky_relu_alpha_x_pattern, bias_add_pattern],
      ordered_inputs=False)
    layer_matches = []
    matched_layer_set = set()
    layer_matcher = GraphMatcher(activation_relu_pattern)
    for match_result in layer_matcher.match_graph_def(graph_def):
        layer_node = match_result.get_op('layer_pattern')
        if layer_node.name not in matched_layer_set:
            matched_layer_set.add(layer_node.name)
            layer_matches.append(match_result)

    layer_matcher = GraphMatcher(activation_leakyrelu_pattern)
    for match_result in layer_matcher.match_graph_def(graph_def):
        layer_node = match_result.get_op('layer_pattern')
        if layer_node.name not in matched_layer_set:
            matched_layer_set.add(layer_node.name)
            layer_matches.append(match_result)

    final_layer_matcher = GraphMatcher(bypass_pattern)
    for match_result in final_layer_matcher.match_graph_def(graph_def):
        layer_node = match_result.get_op('layer_pattern')
        if layer_node.name not in matched_layer_set:
            matched_layer_set.add(layer_node.name)
            layer_matches.append(match_result)

    final_layer_matcher = GraphMatcher(bias_add_pattern)
    for match_result in final_layer_matcher.match_graph_def(graph_def):
        layer_node = match_result.get_op('layer_pattern')
        if layer_node.name not in matched_layer_set:
            matched_layer_set.add(layer_node.name)
            layer_matches.append(match_result)

    return layer_matches