classdef perceptronObject
    properties
        perceptron_size;
        activation_function;
        activation_function_1;                                                                                             %first derivative of the activation function
        activation_input;
        learning_rate;
        momentum_rate;
        weight_vector;
        error;
        prev_momentum;
        input_vect;
        del_in;
        del_out;
        output;
    end
    methods
        function returnObj = perceptronObject(perceptron_size, learning_rate, momentum_rate, activation_function, activation_function_1)  %construct the perceptron object
            returnObj.perceptron_size       = perceptron_size;
            returnObj.activation_function   = activation_function;                                                         %get the handle for the activation function
            returnObj.learning_rate         = learning_rate;                                                               %initialize the learning rate
            returnObj.momentum_rate         = momentum_rate;                                                               %initialize the momentum rate
            returnObj.activation_function_1 = activation_function_1;                                                       %get the handle for the first derivative of the actvation function
            returnObj.weight_vector         = rand(returnObj.perceptron_size + 1, 1);                                      %initialize the weight vector
            returnObj.del_out               = zeros(returnObj.perceptron_size + 1, 1);
            returnObj.error                 = 0;
            returnObj.prev_momentum         = zeros(returnObj.perceptron_size + 1, 1);
        end
    end
    methods(Access = public)
        function obj = pulse_forward(obj, inp_vect, expec_op)
            obj.input_vect = [inp_vect' 1]';
            obj = obj.feedForward(obj.input_vect);
            obj.error = expec_op - obj.output;
        end
        function obj = pulse_backward(obj, del_in)
            obj.del_in = del_in;
            obj = obj.backpropagate();
            obj = obj.weightAdjust();
        end
        function obj = feedForward(obj, inp_vect)
            obj.activation_input = obj.weight_vector' * inp_vect;
            obj.output = obj.activation_function(obj.activation_input);
        end
        function obj = weightAdjust(obj)
            obj.weight_vector = obj.weight_vector + (((obj.learning_rate * obj.del_in *  obj.activation_function_1(obj.activation_input)).* obj.input_vect) + (obj.momentum_rate * obj.prev_momentum));
            obj.prev_momentum = (((obj.learning_rate * obj.del_in *  obj.activation_function_1(obj.activation_input)).* obj.input_vect) + (obj.momentum_rate * obj.prev_momentum));
        end
        function obj = backpropagate(obj)
            obj.del_out = (obj.del_in * obj.activation_function_1(obj.activation_input)).* obj.weight_vector;
        end
    end
end
