% Load the dataset
unzip('cropped_faces.zip');

% Create an imageDatastore and assign labels
imds = imageDatastore('cropped_faces');
imds.Labels = filenames2labels(imds.Files, "ExtractBetween", [1 3]);



% Specify the number of subjects for training (first 40 subjects)
numSubjectsTrain = 40;

% Select the first numSubjectsTrain subjects for training, and the rest for testing
subjects = unique(imds.Labels);
subjectsTrain = subjects(1:numSubjectsTrain);
subjectsTest = subjects(subjectsTrain + 1:end);

% Split the dataset into training and testing based on subjects
imdsTrain = subset(imds, ismember(imds.Labels, subjectsTrain));
imdsTest = subset(imds, ismember(imds.Labels, subjectsTest));

% Load the pretrained VGG19 model

net = vgg19;

% Check the input size of the VGG19 model
inputSize = net.Layers(1).InputSize;

% Replace the classification head with a new one for your task
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

% Use the last 5 images of each subject for testing
imdsTest.Labels = imdsTest.Labels;
augimdsTest = augmentedImageDatastore(inputSize, imdsTest);

% Set training options

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 80, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsTest, ...  % Use test set for validation during training
    'ValidationFrequency', 3, ...
    'ValidationPatience', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%Train the network

netTransfer = trainNetwork(augimdsTrain, layers, options);

% Test the network and extract features

[YPred, scores] = classify(netTransfer, augimdsTest);

YTest = imdsTest.Labels;
accuracy = mean(YPred == YTest);

idx = randperm(numel(imdsTest.Files), 16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTest, idx(i));
    imshow(I)
    label = strcat('Pred: ',cellstr(YPred(idx(i))),' Actual: ',cellstr(YTest(idx(i))));
    title(string(label));
end

layer = 'fc6';
featuresTrain = activations(netTransfer, augimdsTrain, layer, 'OutputAs', 'rows');
featuresTest = activations(netTransfer, augimdsTest, layer, 'OutputAs', 'rows');

% Calculate cosine similarity scores

% Initialize arrays for storing genuine and impostor scores
genuineScores = [];
impostorScores = [];

% Assuming numClasses is 50 as per your dataset
for subject = 1:numClasses
    % Convert categorical labels to double for comparison
    subjectLabel = double(imdsTest.Labels); % Ensure this refers to test labels

    % Extract the indices of all images for this subject
    subjectIndices = find(subjectLabel == subject);
    if numel(subjectIndices) < 4
        continue; 
    end

    % The first image is used for enrollment

    enrollmentIdx = subjectIndices(1);
    enrollmentFeatures = featuresTest(enrollmentIdx, :);

    % The next images are used for verification
    verificationIndices = subjectIndices(2:end);
    verificationFeatures = featuresTest(verificationIndices, :);

    % Calculate cosine similarity scores for each verification image
    similarityScores = 1 - pdist2(enrollmentFeatures, verificationFeatures, 'cosine');

    % The first score is genuine, the rest are impostors
    genuineScores = [genuineScores; similarityScores(1)];
    impostorScores = [impostorScores; similarityScores(2:end)];
end



%Plot testing score distribution histograms

figure;
histogram(genuineScores, 'Normalization', 'probability', 'DisplayName', 'Genuine Scores');
hold on;
histogram(impostorScores, 'Normalization', 'probability', 'DisplayName', 'Impostor Scores');
xlabel('Cosine Similarity Scores');
ylabel('Probability');
title('Testing Score Distribution');
legend;

%  Calculate ROC curve

labels = [ones(size(genuineScores)); zeros(size(impostorScores))];
scores = [genuineScores; impostorScores];
[~, ~, ~, auc] = perfcurve(labels, scores, 1);
[Xroc, Yroc, T, AUC] = perfcurve(labels, scores, 1);
figure; 
plot(Xroc, Yroc);
xlabel('False Positive Rate'); 
ylabel('True Positive Rate');
title(['ROC Curve, AUC = ' num2str(AUC)]);


%  Calculate d' (d-prime)

meanGenuine = mean(genuineScores);
stdGenuine = std(genuineScores);
meanImpostor = mean(impostorScores);
stdImpostor = std(impostorScores);

dPrime = (meanGenuine - meanImpostor) / sqrt((stdGenuine^2 + stdImpostor^2) / 2);

% Display ROC AUC and d' (d-prime)

fprintf('ROC AUC: %.4f\n', auc);
fprintf('d'' (d-prime): %.4f\n', dPrime);

