function [ M ] = train_metric( initial_matrix, training,  mu, pars, k, validation )
    if nargin < 6
        early_stopping = false;
    else
        early_stopping = true;
    end
    M = initial_matrix;
    W = compute_W_from_M(M,k);
    nb_training_constraints = size(training.smaller,1);
    C = 1 / nb_training_constraints;
    old_active_constraints = false(nb_training_constraints,1);
    Kgradient = 0;
    obj = intmax;
    best_obj = intmax;
    best_M = M;
    muW = 0;
    step_size = pars.initial_step;
    best_acc_val = 0;
    iter = 0;
    
    while (true)
        iter = iter + 1;
        if (iter > pars.max_iter)
            break;
        end
        
        if early_stopping
            acc_val = evaluate_metric(M, validation);
            if (acc_val >= best_acc_val)
                best_M_val = M;
            end
        end

        D_ij =sum((training.larger * M) .* training.larger,2);
        D_kl =sum((training.smaller * M) .* training.smaller,2);
        relative_distances = 1 - D_ij + D_kl;
        bool_distances = relative_distances > 0;
        nb_active_constraints = sum(bool_distances);
        new_active_constraints = (~old_active_constraints & bool_distances);
        new_inactive_constraints = (old_active_constraints & ~bool_distances);
        
        
        old_obj = obj;
        obj = C * sum(relative_distances(bool_distances));
        
        if mu
            %W = compute_W_from_M(M,k);
            obj = obj + mu * (sum(sum(W .* M)));
            muW = mu * W;
        end     
        obj = real(obj);
        if (abs(old_obj - obj) < 10^-11)
            break;
        end 
        
        if (step_size < 10^-13)
            break
        end
        if (best_obj > obj)
            best_obj = obj;
            best_M = M;
        end
        if (old_obj < obj)
            step_size = step_size * 0.7;
        else 
            step_size = step_size * 1.02;
        end
      
        
        nb_new_active_constraitns = sum(new_active_constraints) + sum(new_inactive_constraints);
        
        if nb_new_active_constraitns
            if nb_active_constraints < nb_new_active_constraitns
                g1 = training.larger(bool_distances,:);
                g2 = training.smaller(bool_distances,:);
                Kgradient = C.* (g2'*g2 - g1'*g1);
            else
                g3 = training.larger(new_active_constraints,:);
                g4 = training.smaller(new_active_constraints,:);
                g5 = training.larger(new_inactive_constraints,:);
                g6 = training.smaller(new_inactive_constraints,:);
                Kgradient = Kgradient + C.* (g4'*g4 - g3'*g3  - g6'*g6 + g5'*g5);
            end
        end
        
        
        gradient_M = muW + Kgradient;
        M = M - step_size * gradient_M;
        
        [V, D] = eig(M);
        d = real(diag(D));
        d(d < 0) = 0;      
        M = V * diag(d) * V';
        if mu
            W = compute_W_from_eigen(V,d,k);
        end
    end
    M = best_M;
    if early_stopping
        M = best_M_val;
    end
end

function [W] = compute_W_from_M(M,k)
    [V,D] = eig(M);
    [~,b] = sort(diag(D),'ascend');
    V_W = V(:,b(1:k));
    W = V_W * V_W';
end

function [W] = compute_W_from_eigen(V,d,k)
    [~,b] = sort(d,'ascend');
    V_W = V(:,b(1:k));
    W = V_W * V_W';
end
