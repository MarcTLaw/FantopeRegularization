function [mean_accuracy] = split_demo_lfw(split_index)
split_constraints = load('split_constraints.txt');
face_descriptors = load('face_descriptors_sqrt.mat');
X = face_descriptors.X;
X = X-repmat(mean(X,1), size(X,1),1);

max_nb_of_iterations = 200;
nb_total_constraints = size(split_constraints,1);
nb_splits = 10;
split_size = size(split_constraints,1) / nb_splits;
feature_dimension = size(X,2);
mu = 2;
wanted_rank = 40;
test_indices = false(nb_total_constraints,1);
test_indices(((split_index-1)*split_size + 1):(split_index*split_size)) = true;
test_constraints = split_constraints(test_indices,:);
similar_test_constraints = test_constraints(test_constraints(:,3)==1,1:2);
dissimilar_test_constraints = test_constraints(test_constraints(:,3)==-1,1:2);
nb_similar_test_constraints = size(similar_test_constraints,1);
nb_dissimilar_test_constraints = size(dissimilar_test_constraints,1);


training_indices = ~test_indices;
training_constraints = split_constraints(training_indices,:);
similar_training_constraints = training_constraints(training_constraints(:,3)==1,1:2);
dissimilar_training_constraints = training_constraints(training_constraints(:,3)==-1,1:2);
nb_similar_training_constraints = size(similar_training_constraints,1);
nb_dissimilar_training_constraints = size(dissimilar_training_constraints,1);
training_labels = unique(training_constraints(:,1:2));
nb_training_labels = length(training_labels);



for i=1:size(similar_training_constraints,1)
    for j = 1:2
        similar_training_constraints(i,j) = find(similar_training_constraints(i,j)==training_labels);
    end
end

for i=1:size(dissimilar_training_constraints,1)
    for j = 1:2
        dissimilar_training_constraints(i,j) = find(dissimilar_training_constraints(i,j)==training_labels);
    end
end

X_training = X(training_labels,:);
X_training_transpose = X_training';
X_transpose = X';


[COEFF] = princomp(X);
L = (COEFF(:,1:wanted_rank))';
M = L' * L;
W = COEFF*diag([zeros(wanted_rank,1);ones(feature_dimension-wanted_rank,1)])*COEFF';
KappaKimuW = mu * W;

nb_dimensions = length(M);
training_X_similar = zeros(nb_similar_training_constraints,nb_dimensions);
for constraint = 1:nb_similar_training_constraints
    i = similar_training_constraints(constraint,1);
    j = similar_training_constraints(constraint,2);
    x_ij = X_training(i,:) - X_training(j,:);
    training_X_similar(constraint,:) = x_ij;
end
training_X_dissimilar = zeros(nb_dissimilar_training_constraints,nb_dimensions);
for constraint = 1:nb_dissimilar_training_constraints
    i = dissimilar_training_constraints(constraint,1);
    j = dissimilar_training_constraints(constraint,2);
    x_ij = X_training(i,:) - X_training(j,:);
    training_X_dissimilar(constraint,:) = x_ij;
end

test_X_similar = zeros(nb_similar_test_constraints,nb_dimensions);
for constraint = 1:nb_similar_test_constraints
    i = similar_test_constraints(constraint,1);
    j = similar_test_constraints(constraint,2);
    x_ij = X(i,:) - X (j,:);
    test_X_similar(constraint,:) = x_ij;
end
test_X_dissimilar = zeros(nb_dissimilar_test_constraints,nb_dimensions);
for constraint = 1:nb_dissimilar_test_constraints
    i = dissimilar_test_constraints(constraint,1);
    j = dissimilar_test_constraints(constraint,2);
    x_ij = X(i,:) - X (j,:);
    test_X_dissimilar(constraint,:) = x_ij;
end


similar_active_constraints = false(1,nb_similar_training_constraints);
dissimilar_active_constraints = false(1,nb_dissimilar_training_constraints);

margin = 0.5;
obj = intmax;
C = 1;
change_step_size = true;
old_M = M + 10;
best_obj = intmax;
mest_M = 0;
best_L = 0;
step_size = 3*10.^(-5);
disp('beginning training');

for iter = 1:max_nb_of_iterations
    LX_training = L * X_training_transpose;
    
    old_obj = obj;
    obj = 0;
    new_similar_active_constraints = false(1,nb_similar_training_constraints);
    new_dissimilar_active_constraints = false(1,nb_dissimilar_training_constraints);
    new_similar_inactive_constraints = false(1,nb_similar_training_constraints);
    new_dissimilar_inactive_constraints = false(1,nb_dissimilar_training_constraints);
    
    
    for constraint = 1:nb_similar_training_constraints
        i = similar_training_constraints(constraint,1);
        j = similar_training_constraints(constraint,2);
        d_ij = (LX_training(:,i) - LX_training(:,j));
        d_ij = d_ij'*d_ij;
        if d_ij >= (1 - margin)
            obj = obj + (d_ij - 1 + margin);
            if ~similar_active_constraints(constraint)
                new_similar_active_constraints(constraint) = true;
                similar_active_constraints(constraint) = true;
            end
        elseif similar_active_constraints(constraint)
            new_similar_inactive_constraints(constraint) = true;
            similar_active_constraints(constraint) = false;
        end
    end
    
    for constraint = 1:nb_dissimilar_training_constraints
        i = dissimilar_training_constraints(constraint,1);
        j = dissimilar_training_constraints(constraint,2);
        d_ij = (LX_training(:,i) - LX_training(:,j));
        d_ij = d_ij'*d_ij;
        if d_ij <= 1 + margin
            obj = obj + (1 + margin - d_ij);
            if ~dissimilar_active_constraints(constraint)
                new_dissimilar_active_constraints(constraint) = true;
                dissimilar_active_constraints(constraint) = true;
            end
        elseif dissimilar_active_constraints(constraint)
            new_dissimilar_inactive_constraints(constraint) = true;
            dissimilar_active_constraints(constraint) = false;
        end
    end
    obj = C * obj;
    
    if change_step_size
        if (old_obj < obj) && (step_size > 0.00001)
            
            step_size = step_size * 0.8;
            
        else
            step_size = step_size * 1.05;
        end
    end
    
    
    
    nb_new_constraints = sum(new_dissimilar_active_constraints) + sum(new_dissimilar_inactive_constraints) + sum(new_similar_active_constraints) + sum(new_similar_inactive_constraints);
    
    if nb_new_constraints
        nb_active_constraints = sum(similar_active_constraints) + sum (dissimilar_active_constraints);
        
        if nb_active_constraints <= nb_new_constraints
            
            similar_g = training_X_similar(similar_active_constraints,:);
            dissimilar_g = training_X_dissimilar(dissimilar_active_constraints,:);
            Kgradient = C .* (similar_g'*similar_g - dissimilar_g'*dissimilar_g);
            
            
            
            
        else
            
            g1 = training_X_similar(new_similar_active_constraints,:);
            g2 = training_X_dissimilar(new_dissimilar_inactive_constraints,:);
            g3 = training_X_similar(new_similar_inactive_constraints,:);
            g4 = training_X_dissimilar(new_dissimilar_active_constraints,:);
            Kgradient = Kgradient + C.* (g1'*g1 + g2'*g2 - g3'*g3 - g4'*g4);
        end
        
        
        
        
        
        
    end
    if mu
        obj = obj + mu * (sum(sum(W .* M)));
    end
    if best_obj >= obj
        best_obj = obj;
        best_M = M;
        best_L = L;
    end
    
    if ~obj
        disp('break');
        break;
    end
    gradient_M = KappaKimuW + Kgradient;
    %toc
    M = M - step_size * gradient_M;
    
    [V, D] = eig(M);
    
    D(D < 0) = 0;
    
    if mu
        diag_D = diag(D);
        [a,b] = sort(diag_D,'descend');
        V_W = V(:,b((wanted_rank+1):end));
        W = V_W*V_W';
        KappaKimuW = mu * W;
    end
    
    L = sqrt(D) * V';
    
    M = L'*L;
    if (sum(sum(abs(old_M-M))) < 0.0000000001)
        if verbose
            disp('break');
            obj
        end
        break;
    end
end

M = best_M;
L = best_L;

LX = L * X_transpose;
accuracy_similar = 0;
accuracy_dissimilar = 0;

for constraint = 1:nb_similar_test_constraints
    i = similar_test_constraints(constraint,1);
    j = similar_test_constraints(constraint,2);
    d_ij = (LX(:,i) - LX(:,j));
    d_ij = d_ij'*d_ij;
    if d_ij < 1
        accuracy_similar = accuracy_similar + 1;
    end
end



for constraint = 1:nb_dissimilar_test_constraints
    i = dissimilar_test_constraints(constraint,1);
    j = dissimilar_test_constraints(constraint,2);
    d_ij = (LX(:,i) - LX(:,j));
    d_ij = d_ij'*d_ij;
    if d_ij > 1
        accuracy_dissimilar = accuracy_dissimilar + 1;
    end
end

accuracy_similar = 100 * accuracy_similar / nb_similar_test_constraints;
accuracy_dissimilar = 100 * accuracy_dissimilar / nb_dissimilar_test_constraints;
mean_accuracy = (accuracy_similar + accuracy_dissimilar) / 2
end