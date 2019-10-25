#! /usr/bin/env python 3.6 (3379)
#coding=utf-8
# Compiled at: 2019-10-18 19:07:52
#Powered by BugScaner
#http://tools.bugscaner.com/
#如果觉得不错,请分享给你朋友使用吧!
"""
Graph utils

@ author: Sambhav Jain
"""
import sys, re, collections, tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import variable_pb2

def create_node_map(graph_def):
    """Creates a node_map dictionary mapping node names to nodes.
    
    Args:
      graph_def: GraphDef containing nodes
    
    Returns:
      node_map: Dictionary containing an entry indexed by name for every node.
    
    Raises:
      ValueError: If duplicate node names are detected.
    """
    node_map = {}
    for node in graph_def.node:
        if node.name not in node_map.keys():
            node_map[node.name] = node
        else:
            raise ValueError("Duplicate node names detected for '%s'" % node.name)

    return node_map


def create_output_node_map(graph_def):
    """Creates a output_node_map dictionary mapping parent node names to another dictionary
    containing mapping from output (consumer) node names to input index of parent node.
    
    Args:
      graph_def: GraphDef containing nodes
    
    Returns:
      output_node_map: Dictionary containing an entry indexed by name for every node.
    """
    output_node_map = collections.defaultdict(dict)
    for node in graph_def.node:
        for index, input_name in enumerate(node.input):
            input_node_name = node_name_from_input(input_name)
            output_node_map[input_node_name][node.name] = index

    return output_node_map


def get_output_nodes(parent_node, graph_def):
    """Finds all output (consumer) nodes of a parent node in a graphdef.
    
    Args:
      parent_node: NodeDef whose output (consumer) nodes are to be found.
      graph_def: GraphDef containing nodes
    
    Returns:
      output_node_map: Dict mapping output (consumer) node names (as key) and the index of input 
      corresponding to the parent node (as value).
    """
    output_node_map = {}
    for node in graph_def.node:
        for index, input_name in enumerate(node.input):
            input_node_name = node_name_from_input(input_name)
            if input_node_name == parent_node.name:
                output_node_map[node.name] = index

    return output_node_map


def sort_graph_topological(graph_def):
    r"""
    Finds dependencies among graph nodes and sorts them in the correct
    order of execution such that all node dependencies are executed before
    the node itself.
    
    Source
    https://en.wikipedia.org/wiki/Topological_sorting
    
    Reference: Cormen, Thomas H.; Leiserson, Charles E.; Rivest, Ronald L.; Stein, Clifford (2001),
    "Section 22.4: Topological sort", Introduction to Algorithms (2nd ed.),
    MIT Press and McGraw-Hill, pp. 549552, ISBN 0-262-03293-7.
    
    Each node n gets prepended to the output list L only after considering all 
    other nodes which depend on n (all descendants of n in the graph).
    Specifically, when the algorithm adds node n, we are guaranteed that all nodes
    which depend on n are already in the output list L: they were added to L either 
    by the recursive call to _visit() which ended before the call to visit n, or by 
    a call to _visit() which started even before the call to visit n. Since each edge 
    and node is visited once, the algorithm runs in linear time. 
    This depth-first-search-based algorithm is the one described by Cormen et al. (2001); 
    it seems to have been first described in print by Tarjan (1976).
    
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
    
    Args: 
      graph_def: GraphDef containing nodes
    Returns
      L: list of node names sorted in topological order
    """
    global output_node_map
    sys.setrecursionlimit(3000)
    output_node_map = create_output_node_map(graph_def)
    L = []
    unmarked_nodes = []
    temp_marked_nodes = []
    perm_marked_nodes = []
    for node in graph_def.node:
        if node.name not in unmarked_nodes:
            unmarked_nodes.append(node.name)
        else:
            raise ValueError("Duplicate node names detected for '%s'" % node.name)

    while len(unmarked_nodes) != 0:
        node_name = unmarked_nodes.pop(0)
        _visit(node_name, temp_marked_nodes, perm_marked_nodes, L)

    return L


def _visit(node_name, temp_marked_nodes, perm_marked_nodes, L):
    if node_name in perm_marked_nodes:
        return
    if node_name in temp_marked_nodes:
        raise ValueError("Not a DAG (directed-acyclic-graph). May contain cycle at '%s'." % node_name)
    temp_marked_nodes.append(node_name)
    consumer_nodes = output_node_map[node_name]
    for consumer_node_name, _ in consumer_nodes.items():
        _visit(consumer_node_name, temp_marked_nodes, perm_marked_nodes, L)

    perm_marked_nodes.append(node_name)
    L.insert(0, node_name)


def add_static_shapes(graph_def):
    """
    (requires TF > v1.9 to work properly)
    Reinstates missing static shape (_output_shapes attribute) for each tf.NodeDef within a tf.GraphDef
    by loading the stale graph_def within a new session and re-extracting the graph_def with the added
    option "add_shapes=True".
    
    Certain graph transformations involving copying/replacing nodes might lead to missing "_output_shapes"
    attributes for those nodes. Run this function after to fix the missing static shape.
    
    Further, if an input placeholder shape is undefined or missing, then the static shape for any 
    forthcoming nodes is also undefined. In such cases, run the "fix_input_shape" transform followed by
    this function to add the correctly inferred shapes back.
    
    Note: TF version 1.7 has a bug where the static shape for some nodes is not inferred (unknown rank)
    even though the input placeholder shape is defined. This seems to be fixed in TF version 1.9.
    """
    with tf.compat.v1.Session(graph=tf.Graph()) as (sess):
        tf.import_graph_def(graph_def, name='')
        g = tf.compat.v1.get_default_graph()
    output_graph_def = g.as_graph_def(add_shapes=True)
    return output_graph_def


def add_attr_to_tf_node(node, key, name, data):
    """ Add a dictionary 'data' to a TF graph node
    with optional label 'name'. Attr key is 'key'.
    Each element of 'data' should be a tf.AttrValue() instance
    e.g.  a1 = tf.AttrValue(i=8)
          a2 = tf.AttrValue(i=2)
          attrs = {"data1": a1, "data2": a2}
    """
    if name is None:
        node.attr[key].CopyFrom(tf.AttrValue(func={'attr': data}))
    else:
        node.attr[key].CopyFrom(tf.AttrValue(func={'name':name,  'attr':data}))


def restored_variable(name, trainable=True, collections=None, graph=None):
    """
    A variable restored from disk. (FIX for TF>=1.11)
    
    See issue:
    https://github.com/tensorflow/tensorflow/issues/23591
    
    Example:
    1) variable = restored_variable(node.name)
    2) variables = [ restored_variable(node.name, trainable=True, collections=None, graph=g) for node in g.as_graph_def().node if 'Variable' in node.op ]
    """
    variable_def = variable_pb2.VariableDef()
    if graph is None:
        graph = tf.compat.v1.get_default_graph()
    if collections is None:
        collections = [
         tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]
    if trainable:
        if tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES not in collections:
            collections = collections + [tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES]
    variable_def.variable_name = graph.as_graph_element(name).outputs[0].name
    variable_def.snapshot_name = graph.as_graph_element(name + '/read').outputs[0].name
    variable_def.initializer_name = graph.as_graph_element(name + '/Assign').name
    variable_def.trainable = trainable
    i_name = name + '/Initializer/'
    keys = [k for k in graph._nodes_by_name.keys() if k.startswith(i_name) if '/' not in k[len(i_name):]]
    if len(keys) != 1:
        raise ValueError('Could not find initializer for variable', keys)
    variable_def.initial_value_name = graph.as_graph_element(keys[0]).outputs[0].name
    var = tf.Variable.from_proto(variable_def)
    for key in collections:
        graph.add_to_collection(key, var)

    return var


def export_meta_graph(filename, graph_def, ckpt_dir, clear_devices=False, clear_extraneous_savers=False):
    """
    Combines graphdef and metadata (variables, collections) and dumps to a metagraph.
    
    Restores variables into standard collections (global variables, trainable variables)
    before saving out metagraph. CAUTION: Doesn't retain user-defined collections.
    
    This step only exports metagraph; doesn't save out a new ckpt, 
    but saver.save() can export both .meta and .ckpt.
    
    DO NOT run this after 'remove_training_nodes' or 'strip_unused_nodes' since they affect
    the variables' surrounding topology (e.g. read / assign / initialize nodes) which leaves
    the metagraph invalid.
    """
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    ckpt_path = ckpt.model_checkpoint_path
    print(("Using ckpt '{}' to restore variables and collections for 'export_meta_graph'").format(ckpt_path))
    with tf.Session(graph=tf.Graph()) as (sess):
        tf.import_graph_def(graph_def, name='')
        var_list = {}
        reader = tf.train.NewCheckpointReader(ckpt_path)
        for key in reader.get_variable_to_shape_map():
            try:
                tensor = sess.graph.get_tensor_by_name(key + ':0')
            except KeyError:
                continue

            var_list[key] = tensor

        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, ckpt_path)
        variables = [restored_variable(node.name) for node in graph_def.node if 'Variable' in node.op]
        saver.export_meta_graph(filename, clear_devices=clear_devices,
          clear_extraneous_savers=clear_extraneous_savers)


def ensure_graph_is_valid(graph_def):
    """Makes sure that the graph is internally consistent.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/tools/optimize_for_inference_lib.py#L122-L144
    
    Checks basic properties of the graph def and raises an exception if there are
    input references to missing nodes, duplicated names, or other logic errors.
    
    Args:
      graph_def: Definition of a graph to be checked.
    
    Raises:
      ValueError: If the graph is incorrectly constructed.
    """
    node_map = {}
    for node in graph_def.node:
        if node.name not in node_map.keys():
            node_map[node.name] = node
        else:
            raise ValueError("Duplicate node names detected for '%s'" % node.name)

    for node in graph_def.node:
        for input_name in node.input:
            input_node_name = node_name_from_input(input_name)
            if input_node_name not in node_map.keys():
                raise ValueError("Input node '%s' of node '%s' not found." % (input_name, node.name))


def node_name_from_input(node_name):
    """Strips off ports and other decorations to get the underlying node name.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/tools/optimize_for_inference_lib.py#L147-L154
    """
    if node_name.startswith('^'):
        node_name = node_name[1:]
    m = re.search('(.*):\\d+$', node_name)
    if m:
        node_name = m.group(1)
    return node_name


def node_from_map(node_map, name):
    """Pulls a node def from a dictionary for a given name.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/tools/optimize_for_inference_lib.py#L157-L173
    
    Args:
      node_map: Dictionary containing an entry indexed by name for every node.
      name: Identifies the node we want to find.
    
    Returns:
      NodeDef of the node with the given name.
    
    Raises:
      ValueError: If the node isn't present in the dictionary.
    """
    stripped_name = node_name_from_input(name)
    if stripped_name not in node_map:
        raise ValueError("No node named '%s' found in map." % name)
    return node_map[stripped_name]


def values_from_const(node_def):
    """Extracts the values from a const NodeDef as a numpy ndarray.
    
    Reference:
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/tools/optimize_for_inference_lib.py#L176-L194
    
    Args:
      node_def: Const NodeDef that has the values we want to access.
    
    Returns:
      Numpy ndarray containing the values.
    
    Raises:
      ValueError: If the node isn't a Const.
    """
    if node_def.op != 'Const':
        raise ValueError("Node named '%s' should be a Const op for values_from_const." % node_def.name)
    input_tensor = node_def.attr['value'].tensor
    tensor_value = tensor_util.MakeNdarray(input_tensor)
    return tensor_value