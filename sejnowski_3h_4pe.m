perceptron_size = 60;
epoch_reps      = 500;
mu              = 2;
B               = 0.5;
alpha           = 0;
fx              = @(x)(1/(1+exp(-B * x)));
fx_1            = @(x)(B * fx(x)*(1 - fx(x)));

% fx              = @(x)((exp(x) - exp(-x))./(exp(x) + exp(-x)));
% fx_1            = @(x)(1 - (fx(x)).^2);

pe1             = perceptronObject(perceptron_size, mu, alpha, fx, fx_1);
pe2             = perceptronObject(perceptron_size, mu, alpha, fx, fx_1);
pe3             = perceptronObject(perceptron_size, mu, alpha, fx, fx_1);
pe4             = perceptronObject(perceptron_size, mu, alpha, fx, fx_1);

pe21            = perceptronObject(4, mu, alpha, fx, fx_1);
pe22            = perceptronObject(4, mu, alpha, fx, fx_1);
pe23            = perceptronObject(4, mu, alpha, fx, fx_1);
pe24            = perceptronObject(4, mu, alpha, fx, fx_1);

pe31            = perceptronObject(4, mu, alpha, fx, fx_1);
pe32            = perceptronObject(4, mu, alpha, fx, fx_1);
pe33            = perceptronObject(4, mu, alpha, fx, fx_1);
pe34            = perceptronObject(4, mu, alpha, fx, fx_1);
 
pe1_store         = perceptronObject(perceptron_size, mu, alpha, fx, fx_1);
pe2_store         = perceptronObject(perceptron_size, mu, alpha, fx, fx_1);
pe3_store         = perceptronObject(perceptron_size, mu, alpha, fx, fx_1);
pe4_store         = perceptronObject(perceptron_size, mu, alpha, fx, fx_1);

pe21_store        = perceptronObject(4, mu, alpha, fx, fx_1);
pe22_store        = perceptronObject(4, mu, alpha, fx, fx_1);
pe23_store        = perceptronObject(4, mu, alpha, fx, fx_1);
pe24_store        = perceptronObject(4, mu, alpha, fx, fx_1);

pe31_store        = perceptronObject(4, mu, alpha, fx, fx_1);
pe32_store        = perceptronObject(4, mu, alpha, fx, fx_1);
pe33_store        = perceptronObject(4, mu, alpha, fx, fx_1);
pe34_store        = perceptronObject(4, mu, alpha, fx, fx_1);

peo             = perceptronObject(4, mu, alpha, fx, fx_1);
peo_store       = perceptronObject(4, mu, alpha, fx, fx_1);

MSE_T           = zeros(epoch_reps * length(train_set_out), 1);
MSE_V           = zeros(epoch_reps * length(validation_set_out), 1);
op_vect         = zeros(epoch_reps * length(train_set_out), 1);
CMSE_T          = zeros(epoch_reps * length(train_set_out), 1);
CMSE_V          = zeros(epoch_reps * length(validation_set_out), 1);
CMSE_TC         = zeros(epoch_reps, 1);
CMSE_VC         = zeros(epoch_reps, 1);
L               = length(train_set_out);
L2              = length(validation_set_out);
prev_CMSE_V     = 10000000;

for ind_j = 1: 1: epoch_reps
    for ind_i = 1: 1: length(train_set_out)
        pe1 = pe1.pulse_forward(train_set_in(:, ind_i), 0);
        pe2 = pe2.pulse_forward(train_set_in(:, ind_i), 0);
        pe3 = pe3.pulse_forward(train_set_in(:, ind_i), 0);
        pe4 = pe4.pulse_forward(train_set_in(:, ind_i), 0);
        
        pe21 = pe21.pulse_forward([pe1.output pe2.output pe3.output pe4.output]', 0);
        pe22 = pe22.pulse_forward([pe1.output pe2.output pe3.output pe4.output]', 0);
        pe23 = pe23.pulse_forward([pe1.output pe2.output pe3.output pe4.output]', 0);
        pe24 = pe24.pulse_forward([pe1.output pe2.output pe3.output pe4.output]', 0);
        
        pe31 = pe31.pulse_forward([pe21.output pe22.output pe23.output pe24.output]', 0);
        pe32 = pe32.pulse_forward([pe21.output pe22.output pe23.output pe24.output]', 0);
        pe33 = pe33.pulse_forward([pe21.output pe22.output pe23.output pe24.output]', 0);
        pe34 = pe34.pulse_forward([pe21.output pe22.output pe23.output pe24.output]', 0);
       
        peo = peo.pulse_forward([pe31.output pe32.output pe33.output pe34.output]', train_set_out(ind_i));
        
        op_vect(((ind_j - 1) * L ) + ind_i) = peo.output;
        MSE_T(((ind_j - 1) * L ) + ind_i + 1) = MSE_T(((ind_j - 1) * L ) + ind_i) + peo.error^2;
        CMSE_T(((ind_j - 1) * L ) + ind_i) = MSE_T(((ind_j - 1) * L ) + ind_i + 1)/(((ind_j - 1) * L ) + ind_i);
        
        peo = peo.pulse_backward(peo.error);
        
        pe31 = pe31.pulse_backward(peo.del_out(1));
        pe32 = pe32.pulse_backward(peo.del_out(2));
        pe33 = pe33.pulse_backward(peo.del_out(3));
        pe34 = pe34.pulse_backward(peo.del_out(4));
    
        err_sum1 = pe31.del_out + pe32.del_out + pe33.del_out + pe34.del_out;
        
        pe21 = pe21.pulse_backward(err_sum1(1));
        pe22 = pe22.pulse_backward(err_sum1(2));
        pe23 = pe23.pulse_backward(err_sum1(3));
        pe24 = pe24.pulse_backward(err_sum1(4));
        
        err_sum2 = pe21.del_out + pe22.del_out + pe23.del_out + pe24.del_out;
        pe1 = pe1.pulse_backward(err_sum2(1));
        pe2 = pe2.pulse_backward(err_sum2(2));
        pe3 = pe3.pulse_backward(err_sum2(3));
        pe4 = pe4.pulse_backward(err_sum2(4));
    end
    for ind_k = 1: 1: length(validation_set_out)
        pe1 = pe1.pulse_forward(validation_set_in(:, ind_k), 0);
        pe2 = pe2.pulse_forward(validation_set_in(:, ind_k), 0);
        pe3 = pe3.pulse_forward(validation_set_in(:, ind_k), 0);
        pe4 = pe4.pulse_forward(validation_set_in(:, ind_k), 0);

        pe21 = pe21.pulse_forward([pe1.output pe2.output pe3.output pe4.output]', 0);
        pe22 = pe22.pulse_forward([pe1.output pe2.output pe3.output pe4.output]', 0);
        pe23 = pe23.pulse_forward([pe1.output pe2.output pe3.output pe4.output]', 0);
        pe24 = pe24.pulse_forward([pe1.output pe2.output pe3.output pe4.output]', 0);
        
        pe31 = pe31.pulse_forward([pe21.output pe22.output pe23.output pe24.output]', 0);
        pe32 = pe32.pulse_forward([pe21.output pe22.output pe23.output pe24.output]', 0);
        pe33 = pe33.pulse_forward([pe21.output pe22.output pe23.output pe24.output]', 0);
        pe34 = pe34.pulse_forward([pe21.output pe22.output pe23.output pe24.output]', 0);
 
        peo = peo.pulse_forward([pe31.output pe32.output pe33.output pe34.output]', validation_set_out(ind_k));
        
        MSE_V(((ind_j - 1) * L2 ) + ind_k + 1) = MSE_V(((ind_j - 1) * L2 ) + ind_k) + peo.error^2;
        CMSE_V(((ind_j - 1) * L2 ) + ind_k) = MSE_V(((ind_j - 1) * L2 ) + ind_k + 1)/(((ind_j - 1) * L2 ) + ind_k);
        
    end
    if CMSE_V(((ind_j - 1) * L2 ) + ind_k) <= prev_CMSE_V
        prev_CMSE_V = CMSE_V(((ind_j - 1) * L2 ) + ind_k);
        pe1_store = pe1;
        pe2_store = pe2;
        pe3_store = pe3;
        pe4_store = pe4;
  
         pe21_store = pe21;
         pe22_store = pe22;
         pe23_store = pe23;
         pe24_store = pe24;
         
         pe31_store = pe31;
         pe32_store = pe32;
         pe33_store = pe33;
         pe34_store = pe34;

         peo_store = peo;
    end
    CMSE_TC(ind_j) = CMSE_T(((ind_j - 1) * L ) + ind_i);
    CMSE_VC(ind_j) = CMSE_V(((ind_j - 1) * L2 ) + ind_k);
     
end