% Load the dataset
unzip('cropped_faces.zip');

% Create an imageDatastore and assign labels
imds = imageDatastore('cropped_faces');
imds.Labels = filenames2labels(imds.Files, "ExtractBetween", [1 3]);

% Split the dataset into training and validation sets (10 images for training, 5 for validation)
[imdsTrain, imdsValidation] = splitEachLabel(imds, 10, 'randomize');

%  Load the pretrained VGG19 model

net = vgg19;

% Check the input size of the VGG19 model

inputSize = net.Layers(1).InputSize;

% Replace the classification head with a new one

layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));
layers = [layersTransfer;
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20);
    softmaxLayer;
    classificationLayer];

% Set data augmentation and resizing parameters

pixelRange = [-20 20];
imageAugmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);

augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);

%  Set training options

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 80, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 3, ...
    'ValidationPatience', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network

netTransfer = trainNetwork(augimdsTrain, layers, options);

%Test the network and extract features

[YPred,scores] = classify(netTransfer,augimdsValidation);

 YValidation = imdsValidation.Labels;
 accuracy = mean(YPred == YValidation)

idx = randperm(numel(imdsValidation.Files),16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = strcat('Pred: ',cellstr(YPred(idx(i))),' Actual: ',cellstr(YValidation(idx(i))));
    title(string(label));
end

layer = 'fc6';
featuresTrain = activations(netTransfer,augimdsTrain,layer,'OutputAs','rows');
featuresValidation = activations(netTransfer,augimdsValidation,layer,'OutputAs','rows');

% Part 2 - Calculate cosine similarity scores

genuineScores = [];
impostorScores = [];

for subject = 1:numClasses
    subjectLabel = double(imdsValidation.Labels); % Ensure this refers to validation labels
    subjectIndices = find(subjectLabel == subject);
    
    if numel(subjectIndices) < 5
        continue; % Skip if there aren't enough images for this subject
    end
    
    % Enrollment features
    enrollmentIdx = subjectIndices(1);
    enrollmentFeatures = featuresValidation(enrollmentIdx, :);
    
    % Verification features
    verificationIndices = subjectIndices(2:5);
    verificationFeatures = featuresValidation(verificationIndices, :);
    
    % Calculate cosine similarity scores
    similarityScores = 1 - pdist2(enrollmentFeatures, verificationFeatures, 'cosine');
    
    % The first score is genuine, the rest are impostors
    genuineScores = [genuineScores; similarityScores(1)];
    impostorScores = [impostorScores; similarityScores(2:end)'];
end

% Part 3 - Plot testing score distribution histograms

figure;
histogram(genuineScores, 'Normalization', 'probability', 'DisplayName', 'Genuine Scores');
hold on;
histogram(impostorScores, 'Normalization', 'probability', 'DisplayName', 'Impostor Scores');
xlabel('Cosine Similarity Scores');
ylabel('Probability');
title('Testing Score Distribution');
legend;

% Part 4 - Calculate ROC curve and AUC

labels = [ones(size(genuineScores)); zeros(size(impostorScores))];
scores = [genuineScores; impostorScores];
[Xroc, Yroc, ~, AUC] = perfcurve(labels, scores, 1);
figure; 
plot(Xroc, Yroc);
xlabel('False Positive Rate'); 
ylabel('True Positive Rate');
title(['ROC Curve, AUC = ' num2str(AUC)]);

% Part 5 - Calculate d' (d-prime)

meanGenuine = mean(genuineScores);
stdGenuine = std(genuineScores);
meanImpostor = mean(impostorScores);
stdImpostor = std(impostorScores);
dPrime = (meanGenuine - meanImpostor) / sqrt((stdGenuine^2 + stdImpostor^2) / 2);

% Display ROC AUC and d' (d-prime)

fprintf('ROC AUC: %.4f\n', AUC);
fprintf('d'' (d-prime): %.4f\n', dPrime);

% Point on the ROC curve where FAR equals FRR (EER)

eerIndex = find(abs(Xroc - (1 - Yroc)) == min(abs(Xroc - (1 - Yroc))));
eerThreshold = ~(eerIndex);

% PCA Dimensionality Reduction

coeff = pca(featuresValidation);
numDimensionsToKeep = 100; % Choose an appropriate number of dimensions
pcaFeatures = featuresValidation * coeff(:, 1:numDimensionsToKeep);

% Use pcaFeatures for training and validation

% Retrain the model or update the necessary components
netTransferPCA = trainNetwork(augimdsTrain, layers, options);
[YPredPCA, scoresPCA] = classify(netTransferPCA, augimdsValidation);

% Compare metrics for the original and PCA-based models

% For example, ROC AUC
[~, ~, ~, aucOriginal] = perfcurve(labels, scores, 1);
[~, ~, ~, aucPCA] = perfcurve(labels, scoresPCA, 1);

fprintf('Original Model ROC AUC: %.4f\n', aucOriginal);
fprintf('PCA Model ROC AUC: %.4f\n', aucPCA);


% Calculate Rank-1 Identification Rate

[~, minIndices] = min(similarityScores, [], 2);  % Find indices of the minimum scores
correctMatchesRank1 = (minIndices == 1);        % Check if the correct match is ranked first
rank1Rate = sum(correctMatchesRank1) / numel(correctMatchesRank1);

% Calculate Rank-5 Identification Rate

[~, sortedIndices] = sort(similarityScores, 2);  % Sort similarity scores
correctMatchesRank5 = any(sortedIndices(:, 1:4) == 1, 2);  % Check if the correct match is among the top five
rank5Rate = sum(correctMatchesRank5) / numel(correctMatchesRank5);

% Display the rates

fprintf('Rank-1 Identification Rate: %.4f\n', rank1Rate);
fprintf('Rank-5 Identification Rate: %.4f\n', rank5Rate);


% Filter scores based on EER threshold

filteredGenuineScores = genuineScores(genuineScores < eerThreshold);
filteredImpostorScores = impostorScores(impostorScores >= eerThreshold);

% Calculate rank-1 and rank-5 identification rates for filtered scores

if ~isempty(filteredGenuineScores) && ~isempty(filteredImpostorScores)
    sortedGenuine = sort(filteredGenuineScores, 'descend');
    sortedImpostor = sort(filteredImpostorScores, 'descend');

    rank1IdentificationRate = sum(sortedGenuine(1) > sortedImpostor) / numel(sortedGenuine);
    rank5IdentificationRate = sum(sortedGenuine(1:min(5, numel(sortedGenuine))) > sortedImpostor) / numel(sortedGenuine);
else
    rank1IdentificationRate = 0;  % Handle the case when there are no scores
    rank5IdentificationRate = 0;  % Handle the case when there are no scores
end

% Display rank-1 and rank-5 identification rates for filtered scores

fprintf('Rank-1 Identification Rate (filtered): %.4f\n', rank1IdentificationRate);
fprintf('Rank-5 Identification Rate (filtered): %.4f\n', rank5IdentificationRate);












