import tensorflow as tf
import graph_nets as gn
import sonnet as sn
from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets import graphs

def create_graph_dicts(batch):
    """
    description: creates dictionary for input nodes in the batch

    :param batch: a batch of tuples (input , target)
    :return: a dictionary that maps nodes to a tensor containing node representations
    """
    input_dict = []

    for (inputs , targets) in batch:
        input_dict.append({"nodes": inputs[:, None]})

    return input_dict

def compute_loss(outputs , targets):
    return tf.losses.softmax_cross_entropy(tf.one_hot(targets) , outputs)

def get_batched_graphs (train_set):
    """
    description: converts inputs in each batch to complete graphs
    :param train_set: training set containing tuples (batch_input , batch_target)
    :return:
    """
    for batch_input , batch_target  in train_set:
        input_dict = create_graph_dicts(batch_input)
        targets = batch_target

        input_dict = utils_tf.data_dicts_to_graphs_tuple(input_dict)
        input_dict = utils_tf.fully_connect_graph_dynamic(input_dict)

        yield input_dict , targets
