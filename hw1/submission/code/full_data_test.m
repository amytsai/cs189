clc;
clear all;
close all;

%load training data
trainData = load('/Users/Amy/Developer/CS189/hw1/data/train.mat');
trainData = trainData.train;
trainSize = size(trainData.images);
m = trainSize(3);

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

%reshape training data
trainFeatures = zeros(m, 784);
for i = [1:m]
    img = trainData.images(:, :, i);
    imgVector = reshape(img, 1, []);
    trainFeatures(i, : ) = imgVector;
end

%train model
model = train(trainData.labels, sparse(trainFeatures), '-s 2 -c .00000027');
%predict using model
prediction = predict(testData.labels, sparse(testFeatures), model);
error = benchmark(prediction, testData.labels)
    
