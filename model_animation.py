from manim import *

from manim_ml.neural_network.layers.math_operation_layer import MathOperationLayer
from manim_ml.neural_network import NeuralNetwork, Convolutional2DLayer, MaxPooling2DLayer, FeedForwardLayer

# Make the specific scene
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_height = 6.0
config.frame_width = 6.0

class CombinedScene(ThreeDScene):
    def construct(self):
        # Add the network
        nn = NeuralNetwork({
                "conv_1": Convolutional2DLayer(1, 8, 5, filter_spacing=0.32, activation_function="ReLU"),
                "maxpool_1": MaxPooling2DLayer(kernel_size=3),
                "conv_l1": Convolutional2DLayer(3, 8, 3, filter_spacing=0.32, activation_function="ReLU"),
                "conv_l2": Convolutional2DLayer(3, 16, 3, filter_spacing=0.32, activation_function="ReLU"),
                "conv_l3": Convolutional2DLayer(3, 32, 3, filter_spacing=0.32, activation_function="ReLU"),
                "conv_l4": Convolutional2DLayer(3, 64, 3, filter_spacing=0.32, activation_function="ReLU"),
                "feed_forward": FeedForwardLayer(6)
            },
            layer_spacing=0.38
        )
        # Make connections
        input_blank_dot = Dot(
            nn.input_layers_dict["conv_l1"].get_left() - np.array([0.65, 0.0, 0.0])
        )
        nn.add_connection(input_blank_dot, "conv_l1", arc_direction="straight")
        nn.add_connection("conv_l1", "conv_l2")
        output_blank_dot = Dot(
            nn.input_layers_dict["conv_l2"].get_right() + np.array([0.65, 0.0, 0.0])
        )
        nn.add_connection("conv_l2", output_blank_dot, arc_direction="straight")
        # Center the nn
        nn.move_to(ORIGIN)
        self.add(nn)
        # Make code snippet
        # Group it all
        # Play animation
        forward_pass = nn.make_forward_pass_animation()
        self.play(forward_pass)
