clc
clear all
close all

%load training data
trainData = load('/Users/Amy/Developer/CS189/hw1/data/train_small.mat');
trainData = trainData.train;

%load and reshape testing data
testData = load('/Users/Amy/Developer/CS189/hw1/data/test.mat');
testData = testData.test;
imagesize = size(testData.images);
n = imagesize(3);
testFeatures = zeros(n, 784);
for i = [1:n]
    img = testData.images(:, :, i);
    imgVector = reshape(img, 1, []);
    testFeatures(i, :) = imgVector;
end
    
%loop through all small training sets
set = 1;
setSizes = [100 200 500 1000 2000 5000 10000];
setError = zeros(1,7);
for set = [1:7]
    %reshape training data
    trainFeatures = zeros(setSizes(set), 784);
    for i = [1:setSizes(set)]
        img = trainData{set}.images(:, :, i);
        imgVector = reshape(img, 1, []);
        trainFeatures(i, : ) = imgVector;
    end
    
    %train model
    model = train(trainData{set}.labels, sparse(trainFeatures), '-s 2 -c .00000027');
    %predict using model
    prediction = predict(testData.labels, sparse(testFeatures), model);
    %claculate error
    setError(1, set) = benchmark(prediction, testData.labels)
    
    
end

%plot
plot(setSizes, setError);
