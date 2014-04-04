clc
clear all
close all

%load training data
trainData = load('/Users/Amy/Dropbox/CS189/hw3/data/train_small.mat');
trainData = trainData.train;

%load and reshape testing data
testData = load('/Users/Amy/Dropbox/CS189/hw3/data/test.mat');
testData = testData.test;
imagesize = size(testData.images);
n = imagesize(3);
testFeatures = zeros(n, 784);

testLabels = testData.labels;

for i = [1:n]
    img = testData.images(:, :, i);
    imgVector = reshape(img, 1, []);
    testFeatures(i, :) = imgVector;
end

for i = 1:n
    row = testFeatures(i, :);
    testFeatures(i, :)= row/norm(row);
end
    
%loop through all small training sets
setSizes = [100 200 500 1000 2000 5000 10000];
setError = zeros(1,7);
set_mu = zeros(7, 10, 784);
for set = [1:7]
    %reshape training data
    trainFeatures = zeros(setSizes(set), 784);
    labels = trainData{set}.labels;
    
    for i = [1:setSizes(set)]
        img = trainData{set}.images(:, :, i);
        imgVector = reshape(img, 1, []);
        trainFeatures(i, : ) = imgVector;
    end
    
    mu_digits = [];
    sigma_digits = zeros(10, 784, 784);
    t = tabulate(labels);
    
    % sort data
    for i = 0:9
        slice = trainFeatures([labels == i], :);
        %normalize
        slice = slice/norm(slice);
        %fit gaussian
        mu = mean(slice);
        sigma = cov(slice);
        mu_digits = [mu_digits; mu];
        sigma_digits(i+1, :, :) = sigma;
    end
    set_mu(set, :, :) = mu_digits;
    
    %calculate priors

    priors = t(:, 2) / setSizes(set);
    
    %sigma overall

    error = 0;
    probabilities = zeros(10000, 10);
        for class = 0:9
            sigma = sigma_digits(class+1);
            sigma= 6.9*(sigma + eye(784)*.0047);
            g = mvnpdf(testFeatures, mu_digits(class+1, :), sigma);
            probabilities(:, class+1) = g.*priors(class+1);
        end
        %predict
        max_pr = max(probabilities, [], 2);
    
    for i = 1:10000
        prediction = find([probabilities(i, :) == max_pr(i)]) -1;
        if(prediction ~= testLabels(i))
            error = error + 1;
        end 
    end
    setError(set) = error/10000

    
end

%plot

scatter(setSizes, setError);
