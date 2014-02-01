clc
clear all
close all

%Input parameters
start = .0000002;
step = .00000001;
stop = .0000003;
k = 10;


%defining train_small set sizes
set = 7;
setSizes = [100 200 500 1000 2000 5000 10000];

%load training data and reshape
trainData = load('/Users/Amy/Developer/CS189/hw1/data/train_small.mat');
trainData = trainData.train;
trainFeatures = zeros(setSizes(set), 784);
for i = [1:setSizes(set)]
    img = trainData{set}.images(:, :, i);
    imgVector = reshape(img, 1, []);
    trainFeatures(i, :) = imgVector;
end

%load and reshape testing data
testData = load('/Users/Amy/Developer/CS189/hw1/data/test.mat');
testData = testData.test;
imagesize = size(testData.images);
testSize = imagesize(3);
testFeatures = zeros(testSize, 784);
for i = [1:n]
    img = testData.images(:, :, i);
    imgVector = reshape(img, 1, []);
    testFeatures(i, :) = imgVector;
end

%train using k-fold cross validation using different c values.
%.00000029
cost = start;
n = (stop-start)/step;
TestErrors = zeros(n, 1);
costs = zeros(n, 1);
for trial = [1:n]
    CrossErrors = zeros(k, 1);
    indices = crossvalind('Kfold', setSizes(set), k);
    for curK = [1:k]
        options = ['-s 2 -q ', '-c ', num2str(cost)];

        %train using k-1 sets
        model = train(trainData{set}.labels(indices ~= curK, :), sparse(trainFeatures(indices ~= curK, :)), options);

        %predict using remaining set
        prediction = predict(trainData{set}.labels(indices == curK, :), sparse(trainFeatures(indices == curK, :)), model);
        %calculate error
        error = benchmark(prediction, trainData{set}.labels(indices == curK, :));
        CrossErrors(curK) = error;     
    end
    costs(trial) = cost;
    TestErrors(trial) = mean(CrossErrors)
    cost = cost+step; 
end
costs
TestErrors