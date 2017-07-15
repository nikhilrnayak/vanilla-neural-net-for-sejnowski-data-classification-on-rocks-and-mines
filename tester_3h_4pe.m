epoch_reps_t      = 1;
MSE_t             = zeros(epoch_reps_t * length(test_set_out), 1);
op_vect_t         = zeros(epoch_reps_t * length(test_set_out), 1);
CMSE_t            = zeros(epoch_reps_t * length(test_set_out), 1);
rocinps           = zeros(epoch_reps_t * length(test_set_out), 1);
L_t               = length(test_set_out);

pe1               = pe1_store;
pe2               = pe2_store;
pe3               = pe3_store;
pe4               = pe4_store;

pe21               = pe21_store;
pe22               = pe22_store;
pe23               = pe23_store;
pe24               = pe24_store;

pe31               = pe31_store;
pe32               = pe32_store;
pe33               = pe33_store;
pe34               = pe34_store;

peo             = peo_store;

for ind_j = 1: 1: epoch_reps_t
    for ind_i = 1: 1: length(test_set_out)
        pe1 = pe1.pulse_forward(test_set_in(:, ind_i), 0);
        pe2 = pe2.pulse_forward(test_set_in(:, ind_i), 0);
        pe3 = pe3.pulse_forward(test_set_in(:, ind_i), 0);
        pe4 = pe4.pulse_forward(test_set_in(:, ind_i), 0);
        
        pe21 = pe21.pulse_forward([pe1.output pe2.output pe3.output pe4.output]', 0);
        pe22 = pe22.pulse_forward([pe1.output pe2.output pe3.output pe4.output]', 0);
        pe23 = pe23.pulse_forward([pe1.output pe2.output pe3.output pe4.output]', 0);
        pe24 = pe24.pulse_forward([pe1.output pe2.output pe3.output pe4.output]', 0);
        
        pe31 = pe31.pulse_forward([pe21.output pe22.output pe23.output pe24.output]', 0);
        pe32 = pe32.pulse_forward([pe21.output pe22.output pe23.output pe24.output]', 0);
        pe33 = pe33.pulse_forward([pe21.output pe22.output pe23.output pe24.output]', 0);
        pe34 = pe34.pulse_forward([pe21.output pe22.output pe23.output pe24.output]', 0);
        
        peo = peo.pulse_forward([pe31.output pe32.output pe33.output pe34.output]', test_set_out(ind_i));
        
        rocinps(((ind_j - 1) * L_t ) + ind_i) = peo.activation_input;
        op_vect_t(((ind_j - 1) * L_t ) + ind_i) = peo.output;
        MSE_t(((ind_j - 1) * L_t ) + ind_i + 1) = MSE_t(((ind_j - 1) * L_t ) + ind_i) + peo.error^2;
        CMSE_t(((ind_j - 1) * L_t ) + ind_i) = MSE_t(((ind_j - 1) * L_t ) + ind_i)/(((ind_j - 1) * L_t ) + ind_i);
    end
end