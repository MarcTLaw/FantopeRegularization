function [ accuracy ] = evaluate_metric( M, data)
    D_ij =sum((data.smaller * M) .* data.smaller,2);
    D_kl =sum((data.larger * M) .* data.larger,2);
    relative_distances = D_ij - D_kl;
    bool_distances = relative_distances < 0;
    accuracy = mean(bool_distances) * 100;
end

