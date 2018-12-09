import  tensorflow as tf
import sonnet as sn
from graph_nets import blocks


class RRN (sn.AbstractModule):
    """
    Recurrent Relational Networks (https://arxiv.org/abs/1711.08028)
    """

    def __init__(self, name, edge_model_fn, node_model_fn, global_model_fn):
        """
        description: initializes the model

        :param edge_model_fn: Function passed to the edge block, in this paper, it is an MLP
        :param node_model_fn: Function passed to the node block, in this paper for the task of bAbI, it is an LSTM over timesteps
        :param edge_model_fn: Function passed to the global block, in this paper, it is an MLP
        """

        super (RRN , self).__init__(name=name)

        self._edge_block = blocks.EdgeBlock(
            edge_model_fn=edge_model_fn,
            use_edges=False,
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=False
        )

        self._global_block = blocks.GlobalBlock(
            global_model_fn=global_model_fn,
            use_edges=False,
            use_nodes = True,
            use_globals= False,
            nodes_reducer= tf.unsorted_segment_sum
        )

        self._node_model_fn = node_model_fn

    def _build(self, labels, graph, num_steps):
        """
        description: Updates each node according to its label, previous state and neighbours

        first, it passes concatenation of states of each adjacent node to the
        :param labels: Embedding of each node [n_nodes,embedding_length]
        :param graph: GraphTuple containing connectivity information between nodes
        via the senders and receivers fields.

        :return ret_graph: Graph after one step of message passing
        """

        ret_graph = graph

        #lstm for updating nodes during each time step
        lstm = sn.LSTM (2*labels.shape[1])
        state = lstm.initial_state(labels.shape[0])

        for _ in range (num_steps):
            #passing sender and receiver nodes through an MLP
            ret_graph = self._edge_block(ret_graph)

            #aggregating edges to nodes (summing up received edges per node)
            received_edges_aggregator =  blocks.ReceivedEdgesToNodesAggregator(reducer=tf.unsorted_segment_sum)
            messages = received_edges_aggregator(ret_graph)

            #concatenating messages and labels for each node and then passing the result through an MLP
            hidden= self._node_model_fn(tf.concat (labels , messages , axis=1))

            #passing hidden and state through an LSTM
            hidden ,state = lstm (hidden , state)

            #aggregating nodes to global representation
            ret_graph = self._global_block (ret_graph.replace (nodes = hidden))

        return ret_graph



