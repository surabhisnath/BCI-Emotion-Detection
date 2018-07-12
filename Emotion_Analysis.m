clear;
load matrix_new;
load chanlocs;

global x;

setGlobalx(1);
% rng(6)

% -------------------------------------
% clear
% clc
% 
% myFolder = 'E:/Dutyman/Bollywood/bhayanakam_bolly';
% cd(myFolder);
% filePattern = fullfile(myFolder, '*.mat');
% matFiles = dir(filePattern);
% spectra = zeros(20,128,126);
% for k = 1:length(matFiles)
% 	matFilename = fullfile(myFolder, matFiles(k).name);
%     matData = load(matFilename);
%     spectra(k,:,:) = spectopo(matData.bhayanakam_Segment_1,0,250);
% end
% 
% cd ..;
% cd('E:/Dutyman/Bollywood/bhayanakam_bolly_freqdom');
% save('spectra', 'spectra');



% clear
% clc
% 
% myFolder = 'E:/Dutyman/Bollywood';
% 
% filePattern = fullfile(myFolder, '*_bolly_freqdom');
% matFiles = dir(filePattern);
% 
% 
% matrix = zeros(9,20,128,126);
% for k = 1:9
%     cd 'E:/Dutyman/Bollywood';
%     cd(matFiles(k).name);
%     x = load('spectra.mat');
%     matrix(k,:,:,:) = x.spectra;
% end
% 
% 
% 
% 
% matrix_new = zeros(9,20,128,3);
% for i = 1:9
%     for j = 1:20
%         obtain = matrix(i,j,:,:);
%         obtain = squeeze(obtain);
%         theta = [6:9];
%         alpha = [10:14];
%         beta = [15:31];
%         
%         matrix_new(i,j,:,1) = median(obtain(:,theta),2); 
%         matrix_new(i,j,:,2) = median(obtain(:,alpha),2);
%         matrix_new(i,j,:,3) = median(obtain(:,beta),2);
%     end
% end
% 
% save('matrix_new', 'matrix_new');
% ---------------------------------------------------------------

% matrix_new - 9 x 20 x 128 x 3

total = 180;
total_person = 20;
total_emotion = 9;

theta = matrix_new(:,:,:,1);
alpha = matrix_new(:,:,:,2);
beta = matrix_new(:,:,:,3);

resized_alpha_emotion = zeros(180,128); % first 20 rows are one emotion (for all people), next 20 are next emotion..
resized_beta_emotion = zeros(180,128);
resized_theta_emotion = zeros(180,128);

resized_alpha_person = zeros(180,128);  % first 9 rows are one person (all emotions of that person), next 9 are next person..
resized_beta_person = zeros(180,128);
resized_theta_person = zeros(180,128);

for i = 1:9
    resized_alpha_emotion(total_person*(i-1)+1:total_person*(i-1)+total_person,:) = alpha(i,:,:);
    resized_beta_emotion(total_person*(i-1)+1:total_person*(i-1)+total_person,:) = beta(i,:,:);
    resized_theta_emotion(total_person*(i-1)+1:total_person*(i-1)+total_person,:) = theta(i,:,:);
end

for i = 1:20
    resized_alpha_person(total_emotion*(i-1)+1:total_emotion*(i-1)+total_emotion,:) = alpha(:,i,:);
    resized_beta_person(total_emotion*(i-1)+1:total_emotion*(i-1)+total_emotion,:) = beta(:,i,:);
    resized_theta_person(total_emotion*(i-1)+1:total_emotion*(i-1)+total_emotion,:) = theta(:,i,:);
end

% ----------------------------------------------------------------

% Perform normalization

% medians_emotion = median(resized_alpha_emotion')';
% medians_person = median(resized_alpha_person')';
% tosub1 = repmat(medians_emotion,1,128);
% tosub2 = repmat(medians_person,1,128);
% resized_alpha_emotion = resized_alpha_emotion - tosub1;
% resized_alpha_person = resized_alpha_person - tosub2;

% -----------------------------------------------------------------------
% SUPERVIZED

% Divide into test and train

numtrain = 145;
numtest = 35;
class_labels = [1,2,3,4,5,6,7,8,9];
% class_labels = [1,2,2,1,2,2,1,1,1];
% class_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];

% ----------------------------------------------------------------

% train = matrix_new(:,1:numtrain,:,:);
% test = matrix_new(:,numtrain+1:total_,:,:);
% train_theta = train(:,:,:,1);
% train_alpha = train(:,:,:,2);
% train_beta = train(:,:,:,3);
% test_theta = test(:,:,:,1);
% test_alpha = test(:,:,:,2);
% test_beta = test(:,:,:,3);
% train_Y = repmat(class_labels,numtrain,1);
% train_Y = train_Y(:);
% test_Y = repmat(class_labels,numtest,1);
% test_Y = test_Y(:);
% 
% resized_train_alpha = zeros(9*numtrain,128);
% resized_test_alpha = zeros(9*numtest,128);
% for i = 1:9
%     resized_train_alpha(i:i+numtrain-1,:) = train_alpha(i,:,:);
%     resized_test_alpha(i:i+numtest-1,:) = test_alpha(i,:,:);
% end
% 
% YNB = mnrfit(resized_train_alpha, train_Y);
% prediction = predict(YNB, resized_test_alpha);
% prediction
% test_Y
% Accuracy = (nnz(prediction==test_Y)/(numtest*9))*100

% ------------------------------------------------------------------
% Try using random sampling to collect test and train data

% labels = repmat(class_labels,total_person,1);
% % labels = repmat(class_labels,total_emotion,1);
% labels = labels(:);
% temp_resized = [labels,resized_alpha_emotion];
% % temp_resized = [labels,resized_theta_person];

% indices = randperm(total,numtest);
% train_set_X = temp_resized(setdiff([1:total],indices),:);
% train_set_Y = train_set_X(:,1);
% train_set_X = train_set_X(:,2:end);
% test_set_X = temp_resized(indices,:);
% test_set_Y = test_set_X(:,1);
% test_set_X = test_set_X(:,2:end);
% 
% YSVM = fitctree(train_set_X, train_set_Y);
% prediction = predict(YSVM,test_set_X);
% Accuracy = (nnz((prediction==test_set_Y))/numtest)*100

% ---------------------------------------------------------------------

% Try supervized by taking 179 train, 1 test, do this 180 times

% outputs_obtained = zeros(180,1);
% outputs_actual = labels;
% train_set_X = temp_resized_alpha(2:180,:);
% train_set_Y = train_set_X(:,1);
% train_set_X = train_set_X(:,2:end);
% NBStruct = fitcnb(train_set_X,train_set_Y);
% outputs_obtained(1,1) = predict(NBStruct, resized_alpha(1,:));
% 
% for i = 2:180
%     train_set_X = [temp_resized_alpha(1:i-1,:);temp_resized_alpha(i+1:180,:)];
%     train_set_Y = train_set_X(:,1);
%     train_set_X = train_set_X(:,2:end);
%     NBStruct = fitcnb(train_set_X,train_set_Y);
%     outputs_obtained(i,1) = predict(NBStruct, resized_alpha(i,:));
% end
% 
% % Hopeless, NB gives 6% accuracy
% Accuracy = (nnz((outputs_obtained==outputs_actual))/180)*100

% --------------------------------------------------------------

% Leave one subject out

% outputs_obtained = zeros(180,1);
% labels = [1;2;3;4;5;6;7;8;9];
% labels = repmat(labels,1,20);
% labels = labels(:);
% outputs_actual = labels;
% temp_resized = [labels, resized_alpha_person];
% 
% for i = 1:20
%     train_set_X = [temp_resized(1:(9*(i-1)),:);temp_resized(9*i+1:180,:)];
%     train_set_Y = train_set_X(:,1);
%     train_set_X = train_set_X(:,2:end);
%     NBStruct = fitcnb(train_set_X,train_set_Y);
%     outputs_obtained(((i-1)*9+1):(i*9),:) = predict(NBStruct, resized_alpha_person(9*(i-1)+1:9*i,:));
% end
% 
% Accuracy = (nnz((outputs_obtained==outputs_actual))/180)*100

% --------------------------------------------------------------

% Leave one emotion out

outputs_obtained = zeros(180,1);
labels = [1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20];
labels = repmat(labels,1,9);
labels = labels(:);
outputs_actual = labels;
temp_resized = [labels, resized_alpha_person];

for i = 1:9
    train_set_X = [temp_resized(1:(20*(i-1)),:);temp_resized(20*i+1:180,:)];
    train_set_Y = train_set_X(:,1);
    train_set_X = train_set_X(:,2:end);
    NBStruct = fitcnb(train_set_X,train_set_Y);
    outputs_obtained(((i-1)*20+1):(i*20),:) = predict(NBStruct, resized_alpha_person(20*(i-1)+1:20*i,:));
end

Accuracy = (nnz((outputs_obtained==outputs_actual))/180)*100


% --------------------------------------------------------------------
% UNSUPERVISED

% K-means and K-medoids
orig_resized_alpha = resized_alpha_person;

% plot(alpha_idx);
% beta_idx = kmeans(resized_beta_emotion,2,'Distance','correlation');
% theta_idx = kmeans(resized_theta_emotion,2,'Distance','correlation');
% alpha_idx1 = kmedoids(resized_alpha_emotion,2,'Distance','spearman');
% beta_idx1 = kmedoids(resized_beta_emotion,2,'Distance','spearman');
% theta_idx1 = kmedoids(resized_theta_emotion,2,'Distance','spearman');


answerclass = zeros(1,180); %row matrix
sizes = zeros(1,1);

% original_resized_alpha = resized_alpha;
% original_resized_alpha = resized_beta;
% original_resized_alpha = resized_theta;
% 
% % Try by changing order (switch Bhayanakam and Hasyam)- VERY SURPRISING, STILL DOES SAME GROUPING
% resized_alpha = [resized_alpha(1:20,:);resized_alpha(61:80,:);resized_alpha(41:60,:);resized_alpha(21:40,:);resized_alpha(81:180,:)];
 
sno = [1:180]';
resized_alpha_person = [sno,resized_alpha_person];
resized_beta_person = [sno,resized_beta_person];
resized_theta_person = [sno,resized_theta_person];

centres = zeros(1,128);
% [answerclass,sizes] = performKmeans(resized_alpha_person, answerclass, sizes);
[answerclass,sizes,centres] = performKmeans(resized_alpha_person, answerclass, sizes, zeros(1,128), centres);
getGlobalx()-1
num_clusters = getGlobalx()-1;

% -----------------------------
% perform merging

new_sizes = sizes;
redundant_clusters = find(sizes<6);
acc_num_clusters = num_clusters - size(redundant_clusters,2);
acceptable_clusters = find(sizes>=6);
acceptable_centres = [[1:num_clusters]',centres];
acceptable_centres = acceptable_centres(acceptable_clusters,:);
new_sizes(redundant_clusters) = 0;

for m = 1:size(redundant_clusters,2)
    dist = zeros(1,size(acceptable_clusters,2));
    res = find(answerclass==redundant_clusters(1,m));
    for b = 1:size(res,2)
        for n = 1:size(acceptable_clusters,2)
            dist(1,n) = norm(resized_alpha_person(res(1,b),2:129)-acceptable_centres(n,2:129)); 
        end
        [minimum,ind] = min(dist);
        answerclass(res(1,b)) = acceptable_centres(ind,1);
        new_sizes(acceptable_centres(ind,1)) = new_sizes(acceptable_centres(ind,1)) + 1;
    end
end

% -----------------------------

reshaped_answerclass_person = reshape(answerclass,9,20);
[mode_person, freq_person] = mode(reshaped_answerclass_person);
reshaped_answerclass_emotion = reshaped_answerclass_person';
[mode_emotion, freq_emotion] = mode(reshaped_answerclass_emotion);


% for alpha - 16 clusters with seed 4
% for beta - 22 clusters with seed 4
% for theta - 20 clusters with seed 4

% Confusion Matrix
confmat_emotion = zeros(9,num_clusters);
for i = 1:9
    for j = 1:20
        get = reshaped_answerclass_emotion(j,i);
        confmat_emotion(i,get) = confmat_emotion(i,get)+1; 
    end
end

confmat_person = zeros(20,num_clusters);
for i = 1:9
    for j = 1:20
        get = reshaped_answerclass_person(i,j);
        confmat_person(j,get) = confmat_person(j,get)+1; 
    end
end


cluster_info = zeros(num_clusters,1);

for u = 1:num_clusters
    cluster_info(u,1:size(find(answerclass==u),2)) = find(answerclass==u);
    hist(u) = size(find(answerclass==u),2);
end

% figure;
% histogram(hist)
% figure;
% bar(hist)

for u = 1:acc_num_clusters
    figure;
    topoplot(acceptable_centres(u,2:129),chanlocs);
    colorbar;
end


% alpha_idx = kmedoids(orig_resized_alpha,num_clusters);
% ans = find(alpha_idx==1);
% ans = reshape(alpha_idx,9,20);


% Cannot form a good diagonal if repetition

% new_confmat_emotion = [confmat_emotion(:,11),confmat_emotion(:,10),confmat_emotion(:,14),confmat_emotion(:,9),confmat_emotion(:,1),confmat_emotion(:,3),confmat_emotion(:,2),confmat_emotion(:,5),confmat_emotion(:,4),confmat_emotion(:,6),confmat_emotion(:,7),confmat_emotion(:,8),confmat_emotion(:,12),confmat_emotion(:,13),confmat_emotion(:,15),confmat_emotion(:,16)];
% new_confmat_person = [confmat_person(:,3),confmat_person(:,14),confmat_person(:,10),confmat_person(:,4),confmat_person(:,1),confmat_person(:,2),confmat_person(:,16),confmat_person(:,5),confmat_person(:,7),confmat_person(:,9),confmat_person(:,6),confmat_person(:,8),confmat_person(:,11),confmat_person(:,12),confmat_person(:,13),confmat_person(:,15)];
% new_confmat_emotion = [confmat_emotion(:,13),confmat_emotion(:,14),confmat_emotion(:,17),confmat_emotion(:,8),confmat_emotion(:,2),confmat_emotion(:,1),confmat_emotion(:,3:7),confmat_emotion(:,9:12),confmat_emotion(:,15:16),confmat_emotion(:,18:22)];
% new_confmat_person = [confmat_person(:,8),confmat_person(:,2),confmat_person(:,9),confmat_person(:,18),confmat_person(:,17),confmat_person(:,22),confmat_person(:,1),confmat_person(:,6),confmat_person(:,3),confmat_person(:,4),confmat_person(:,5),confmat_person(:,7),confmat_person(:,10:16),confmat_person(:,19:21)];
% new_confmat_emotion = [confmat_emotion(:,16),confmat_emotion(:,18),confmat_emotion(:,12),confmat_emotion(:,8),confmat_emotion(:,9),confmat_emotion(:,1),confmat_emotion(:,5),confmat_emotion(:,2:4),confmat_emotion(:,6:7),confmat_emotion(:,10:11),confmat_emotion(:,13:15),confmat_emotion(:,17),confmat_emotion(:,19:20)];
% new_confmat_person = [confmat_person(:,12),confmat_person(:,9),confmat_person(:,5),confmat_person(:,18),confmat_person(:,10),confmat_person(:,8),confmat_person(:,19),confmat_person(:,6),confmat_person(:,1),confmat_person(:,2),confmat_person(:,3:4),confmat_person(:,7),confmat_person(:,11),confmat_person(:,13:17),confmat_person(:,20)];

% -----------------------------------------------------------------


% EEG Spectra for all 9 emotions for person 2 and person 3 (Very similar trend)
% figure;
% plot(resized_alpha(2,:)); hold on;
% plot(resized_alpha(12,:)); hold on;
% plot(resized_alpha(22,:)); hold on;
% plot(resized_alpha(32,:)); hold on;
% plot(resized_alpha(42,:)); hold on;
% plot(resized_alpha(52,:)); hold on;
% plot(resized_alpha(62,:)); hold on;
% plot(resized_alpha(72,:)); hold on;
% plot(resized_alpha(82,:));
% figure;
% plot(resized_alpha(3,:)); hold on;
% plot(resized_alpha(13,:)); hold on;
% plot(resized_alpha(23,:)); hold on;
% plot(resized_alpha(33,:)); hold on;
% plot(resized_alpha(43,:)); hold on;
% plot(resized_alpha(53,:)); hold on;
% plot(resized_alpha(63,:)); hold on;
% plot(resized_alpha(73,:)); hold on;
% plot(resized_alpha(83,:));


% temp = find(alpha_idx1==2);
% temp2 = reshape(alpha_idx1,20,9);
% classif = zeros(1,9);
% 
% for i = 1:20
%     classif = classif + (temp2(i,:) == 2);
% end

% ------------------------------------------------------------