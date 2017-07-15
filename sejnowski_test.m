perceptron_size = 60;
mu      = 0.00005;
alpha   = 0.0000;
fx      = @(x)(1/(1+exp(-x)));
fx_1    = @(x)(fx(x)*(1 - fx(x)));
pe1     = perceptronObject(perceptron_size, mu, alpha, fx, fx_1);
pe2     = perceptronObject(perceptron_size, mu, alpha, fx, fx_1);
peo     = perceptronObject(2, mu, alpha, fx, fx_1);
MSE     = zeros(1000 * length(train_set_out), 1);
op_vect = zeros(1000 * length(train_set_out), 1);
CMSE    = zeros(1000 * length(train_set_out), 1);
L       = length(train_set_out);

for ind_j = 1: 1: 10000
    for ind_i = 1: 1: length(train_set_out) - 1
        pe1 = pe1.pulse_forward(train_set_in(:, ind_i), 0);
        pe2 = pe2.pulse_forward(train_set_in(:, ind_i), 0);
        peo = peo.pulse_forward([pe1.output pe2.output]', train_set_out(ind_i));
        op_vect(((ind_j - 1) * L ) + ind_i) = peo.output;
        MSE(((ind_j - 1) * L ) + ind_i + 1) = MSE(((ind_j - 1) * L ) + ind_i) + peo.error^2;
        CMSE(((ind_j - 1) * L ) + ind_i) = MSE(((ind_j - 1) * L ) + ind_i)/ind_i;
        peo = peo.pulse_backward(peo.error);
        pe1 = pe1.pulse_backward(peo.del_out(1));
        pe2 = pe2.pulse_backward(peo.del_out(2));
    end
end