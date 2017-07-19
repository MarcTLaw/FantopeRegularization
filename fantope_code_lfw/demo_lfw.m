nb_splits = 10;
split_accuracy = zeros(nb_splits,1);
for i=1:nb_splits
    fprintf('Preprocessing data for split %d.\n',i);
    split_accuracy(i) = split_demo_lfw(i);
end
accuracy = mean(split_accuracy)
standard_error = std(split_accuracy)/sqrt(nb_splits)