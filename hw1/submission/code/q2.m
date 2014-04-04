clc
clear all
close all

trainData = load('/Users/Amy/Developer/CS189/hw1/data/train_small.mat');
trainData = trainData.train;
set = 7;
setSizes = [100 200 500 1000 2000 5000 10000];
trainFeatures = zeros(setSizes(set), 784);
for i = [1:setSizes(set)]
    img = trainData{set}.images(:, :, i);
    imgVector = reshape(img, 1, []);
    trainFeatures(i, : ) = imgVector;
end

model = train(trainData{set}.labels, sparse(trainFeatures), '-s 4');

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

prediction = predict(testData.labels, sparse(testFeatures), model);
matrix = confusionmat(testData.labels, prediction);
matrix = matrix./max(max(matrix));
imshow(matrix, 'InitialMagnification',6400);
colormap(jet);