
import numpy as np
import tensorflow as tf
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from minedojo.sim import ALL_CRAFT_SMELT_ITEMS, ALL_ITEMS
N_ALL_ITEMS = len(ALL_ITEMS)

class RllibPPOModel(TFModelV2):
    """
    Model that will map environment states to action probabilities. Will be shared across agents
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs
    ):
        super(RllibPPOModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        # params we got to pass in from the call to "run"
        custom_params = model_config["custom_model_config"]
        ## Parse custom network params
        num_hidden_layers = custom_params["NUM_HIDDEN_LAYERS"]
        size_hidden_layers = custom_params["SIZE_HIDDEN_LAYERS"]
        num_filters = custom_params["NUM_FILTERS"]
        num_convs = custom_params["NUM_CONV_LAYERS"]
        d2rl = custom_params["D2RL"]
        assert type(d2rl) == bool

        ## Create graph of custom network. It will under a shared tf scope such that all agents
        ## use the same model

        self.rgb = tf.keras.Input(
            shape = (3, 288, 512), name = "rgb"
        )

        self.equipment = tf.keras.Input(
            shape = (N_ALL_ITEMS,), name = "equipment"
        )

        self.inventory = tf.keras.Input(
            shape = (N_ALL_ITEMS,), name = "inventory"
        )

        self.inventory_delta = tf.keras.Input(
            shape = (N_ALL_ITEMS,), name = "inventory_delta"
        )

        self.inventory_max = tf.keras.Input(
            shape = (N_ALL_ITEMS,), name = "inventory_max"
        )

        self.life_stats = tf.keras.Input(
            shape = (3,), name = "life_stats"
        )

        #self.damage_received = tf.keras.Input(
        #    shape = (1,), name = "damage_received"
        #)

        #self.mask_action_type = tf.keras.Input(
        #    shape = (19,), name = "mask_action_type"
        #)

        #self.mask_craft_smelt = tf.keras.Input(
        #    shape = (len(ALL_CRAFT_SMELT_ITEMS),), name= "mask_craft_smelt"
        #)

        #self.mask_destroy = tf.keras.Input(
        #    shape = (N_ALL_ITEMS,), name = "mask_destroy"
        #)

        #self.mask_equip_place = tf.keras.Input(
        #    shape = (N_ALL_ITEMS,), name = "mask_equip_place"
        #)

        combined_inputs = tf.keras.layers.Concatenate()([self.equipment, self.inventory, self.inventory_delta, self.inventory_max, self.life_stats,])

        self.combined_inputs = combined_inputs
        out1 = self.rgb
        out2 = self.combined_inputs
        # Apply initial conv layer with a larger kenel (why?)
        if num_convs > 0:
            y = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.leaky_relu,
                name="conv_initial",
            )
            out1 = y(out1)

        # Apply remaining conv layers, if any
        for i in range(0, num_convs - 1):
            padding = "same" if i < num_convs - 2 else "valid"
            outa = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=[3, 3],
                padding=padding,
                activation=tf.nn.leaky_relu,
                name="conv_{}".format(i),
            )

            out1 = outa(out1)

        # Apply dense hidden layers, if any
        conv_out1 = tf.keras.layers.Flatten()(out1)
        conv_out2 = tf.keras.layers.Flatten()(out2)
        self.combined_inputs = tf.keras.layers.Concatenate()([conv_out1, conv_out2])
        out1 = self.combined_inputs
        for i in range(num_hidden_layers):
            #if i > 0 and d2rl:
                #out1 = tf.keras.layers.Concatenate()([out1, conv_out1])
            out1 = tf.keras.layers.Dense(size_hidden_layers)(out1)
            out1 = tf.keras.layers.LeakyReLU()(out1)

        # Linear last layer for action distribution logits
        layer_out = tf.keras.layers.Dense(self.num_outputs)(out1)

        # Linear last layer for value function branch of model
        value_out = tf.keras.layers.Dense(1)(out1)

        self.base_model = tf.keras.Model([self.rgb, self.equipment, self.inventory, self.inventory_delta, self.inventory_max, self.life_stats,], [layer_out, value_out])

    def forward(self, input_dict, state=None, seq_lens=None):
        model_out, self._value_out= self.base_model(input_dict["obs"])
        return model_out, state
        #print(model_out)
        #action_type_mask = input_dict["obs"]["mask_action_type"]
        #expanded_mask = tf.expand_dims(action_type_mask, axis=1)
        #tiled_mask = tf.tile(expanded_mask, [1, tf.shape(model_out)[1], 1])
        #reduced_mask = tf.reduce_any(tf.not_equal(tiled_mask, 0.0), axis=2)
        #tiled_mask = tf.transpose(tiled_mask, perm=[0, 2, 1])
        #broadcasted_mask = tf.broadcast_to(action_type_mask, tf.shape(model_out))
        #masked_logits = model_out - (1 - tiled_mask) * 1e9
        #return tf.nn.softmax(masked_logits), state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
