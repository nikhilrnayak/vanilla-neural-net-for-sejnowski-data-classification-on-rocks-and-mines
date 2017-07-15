count = 0;
fparade = 0;
tparade = 0;
fnarade = 0;
tnarade = 0;
for ind_i = 1 : 1: length(test_set_out)
    if (op_vect_t(ind_i) >= 0.5 & test_set_out(ind_i) == 1) | (op_vect_t(ind_i) < 0.5 & test_set_out(ind_i) == 0)
        count = count + 1;
    end
    if (op_vect_t(ind_i) >= 0.5 & test_set_out(ind_i) == 1)
       tparade = tparade + 1;
    end
    if (op_vect_t(ind_i) >= 0.5 & test_set_out(ind_i) == 0)
        fparade = fparade + 1;
    end
    if (op_vect_t(ind_i) < 0.5 & test_set_out(ind_i) == 0)
       tnarade = tnarade + 1;
    end
    if (op_vect_t(ind_i) < 0.5 & test_set_out(ind_i) == 1)
        fnarade = fnarade + 1;
    end
end