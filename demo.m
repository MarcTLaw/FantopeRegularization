clear all;
close all;

%%%%% Setup to create the toy dataset

dimensionality = 100;
target_rank = 15;
k = dimensionality - target_rank;
number_of_samples = 10000;
training_size = 10^4;
validation_size = 10^4;
test_size = 10^4;

fprintf('Creating toy dataset:\n- %d random samples\n- %d training constraints\n- %d validation constraints\n- %d test constraints\n- %dx%d dimensional groundtruth distance matrix T with rank(T) = %d\n', number_of_samples, training_size, validation_size, test_size, dimensionality, dimensionality, target_rank);
[ T, X, training_constraints, validation_constraints, test_constraints ] = create_toy_dataset( dimensionality, target_rank, number_of_samples, training_size, validation_size, test_size );

disp('Preprocessing data');
[ training, validation, test] = preprocess_toy( X, training_constraints, validation_constraints, test_constraints );


% evaluate_metric(T, training)
% evaluate_metric(T, validation)
% evaluate_metric(T, test)




disp(' ');
disp(' ');


%%%%% Training the metric without regularization

M = zeros(dimensionality);


mu = 0;
pars.initial_step = 1;
pars.max_iter = intmax;
disp('Training metric without regularization');
M_train = train_metric(M, training,  mu, pars, k );

disp('End of training');
fprintf('No regularization: test accuracy of %f percent, and %d dominant eigenvalues.\n',evaluate_metric(M_train, test),leading_eigenvalues( M_train ));
%evaluate_metric(M_train, training)
%evaluate_metric(M_train, validation)
%evaluate_metric(M_train, test)



disp(' ');
disp(' ');



%%%%% Training the metric with fantope regularization

mu_values = [10.^([-2:2]), 5*10.^([-2:2])];

disp('Training metric with fantope regularization (and cross validating the regularization parameter)');
best_M_fantope = 0;
best_accuracy_fantope = 0;
for mu = mu_values
    M_fantope = train_metric(M_train, training,  mu, pars, k);
    acc_fantope = evaluate_metric(M_fantope, validation);
    if acc_fantope >= best_accuracy_fantope
        best_M_fantope = M_fantope;
        best_accuracy_fantope = acc_fantope;
    end
end

disp('End of training');
fprintf('Fantope regularization: test accuracy of %f percent, and %d dominant eigenvalues.\n',evaluate_metric(best_M_fantope, test),leading_eigenvalues( best_M_fantope ));




disp(' ');
disp(' ');


%%%%% Training the metric with nuclear norm regularization


disp('Training metric with nuclear norm regularization (and cross validating the regularization parameter)');
best_M_nuclear = 0;
best_accuracy_nuclear = 0;
for mu = mu_values
    M_nuclear = train_metric(M_train, training,  mu, pars, dimensionality);
    acc_nuclear = evaluate_metric(M_nuclear, validation);
    if acc_nuclear >= best_accuracy_nuclear
        best_M_nuclear = M_nuclear;
        best_accuracy_nuclear = acc_nuclear;
    end
end
disp('End of training');
fprintf('Nuclear norm regularization: test accuracy of %f percent, and %d dominant eigenvalues.\n',evaluate_metric(best_M_nuclear, test),leading_eigenvalues( best_M_nuclear ));

disp(' ');
disp(' ');

disp('Results without early stopping');
fprintf('- No regularization: test accuracy of %f percent, and %d dominant eigenvalues.\n',evaluate_metric(M_train, test),leading_eigenvalues( M_train ));
fprintf('- Nuclear norm regularization: test accuracy of %f percent, and %d dominant eigenvalues.\n',evaluate_metric(best_M_nuclear, test),leading_eigenvalues( best_M_nuclear ));
fprintf('- Fantope regularization: test accuracy of %f percent, and %d dominant eigenvalues.\n',evaluate_metric(best_M_fantope, test),leading_eigenvalues( best_M_fantope ));





disp(' ');
disp(' ');

disp('Tests with early stopping');


mu = 0;
initial_step = 1;
disp('Training metric without regularization');
M_train = train_metric(M, training,  mu, pars, k , validation);

disp('End of training');
fprintf('No regularization: test accuracy of %f percent, and %d dominant eigenvalues.\n',evaluate_metric(M_train, test),leading_eigenvalues( M_train ));
%evaluate_metric(M_train, training)
%evaluate_metric(M_train, validation)
%evaluate_metric(M_train, test)

disp(' ');
disp(' ');


disp('Training metric with fantope regularization (and cross validating the regularization parameter)');
best_M_fantope = 0;
best_accuracy_fantope = 0;
for mu = mu_values
    M_fantope = train_metric(M_train, training,  mu, pars, k, validation);
    acc_fantope = evaluate_metric(M_fantope, validation);
    if acc_fantope >= best_accuracy_fantope
        best_M_fantope = M_fantope;
        best_accuracy_fantope = acc_fantope;
    end
end

disp('End of training');
fprintf('Fantope regularization: test accuracy of %f percent, and %d dominant eigenvalues.\n',evaluate_metric(best_M_fantope, test),leading_eigenvalues( best_M_fantope ));

disp(' ');
disp(' ');

disp('Training metric with nuclear norm regularization (and cross validating the regularization parameter)');
best_M_nuclear = 0;
best_accuracy_nuclear = 0;
for mu = mu_values
    M_nuclear = train_metric(M_train, training,  mu, pars, dimensionality, validation);
    acc_nuclear = evaluate_metric(M_nuclear, validation);
    if acc_nuclear >= best_accuracy_nuclear
        best_M_nuclear = M_nuclear;
        best_accuracy_nuclear = acc_nuclear;
    end
end
disp('End of training');
fprintf('Nuclear norm regularization: test accuracy of %f percent, and %d dominant eigenvalues.\n',evaluate_metric(best_M_nuclear, test),leading_eigenvalues( best_M_nuclear ));


disp(' ');
disp(' ');

disp('Results with early stopping');

fprintf('No regularization: test accuracy of %f percent, and %d dominant eigenvalues.\n',evaluate_metric(M_train, test),leading_eigenvalues( M_train ));

fprintf('Nuclear norm regularization: test accuracy of %f percent, and %d dominant eigenvalues.\n',evaluate_metric(best_M_nuclear, test),leading_eigenvalues( best_M_nuclear ));
fprintf('Fantope regularization: test accuracy of %f percent, and %d dominant eigenvalues.\n',evaluate_metric(best_M_fantope, test),leading_eigenvalues( best_M_fantope ));




