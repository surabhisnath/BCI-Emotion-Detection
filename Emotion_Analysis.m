global x;
% while(size(find(alpha_idx1==1),1)>20 | size(find(alpha_idx1==2),1)>20)
%     
%     if size(find(alpha_idx1==1),1)>20 & size(find(alpha_idx1==2),1)>20
%         resized_alpha1 = resized_alpha(find(alpha_idx1==1),1);
%         resized_alpha2 = resized_alpha(find(alpha_idx1==2),1);
%         resized_alpha
%         alpha_idx1 = kmedoids(resized_alpha,2,'Distance','spearman');
%     elseif size(find(alpha_idx1==1),1)>20
%         find(alpha_idx1==2)
%         resized_alpha = resized_alpha(find(alpha_idx1==1),1);
%         alpha_idx1 = kmedoids(resized_alpha,2,'Distance','spearman');
%         
%     elseif size(find(alpha_idx1==2),1)>20
%         store alpha_idx1==1
%         resized_alpha = resized_alpha(surafind(alpha_idx1==2),1);
%         alpha_idx1 = kmedoids(resized_alpha,2,'Distance','spearman');
%     else
%         store both
%     end
% end
setGlobalx(1);


rng(4);
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


% matrix_new - 9 x 20 x 128 x 3

% Divide into test and train

total = 20;
% numtrain = 12;
% numtest = total-numtrain;
% train = matrix_new(:,1:numtrain,:,:);
% test = matrix_new(:,numtrain+1:total,:,:);
% 
% train_theta = train(:,:,:,1);
% train_alpha = train(:,:,:,2);
% train_beta = train(:,:,:,3);
% test_theta = test(:,:,:,1);
% test_alpha = test(:,:,:,2);
% test_beta = test(:,:,:,3);

theta = matrix_new(:,:,:,1);
alpha = matrix_new(:,:,:,2);
beta = matrix_new(:,:,:,3);


class_labels = [1,2,3,4,5,6,7,8,9];
% train_Y = repmat(class_labels,numtrain,1);
% train_Y = train_Y(:);
% test_Y = repmat(class_labels,numtest,1);
% test_Y = test_Y(:);


% resized_train_alpha = zeros(108,128);
% resized_test_alpha = zeros(72,128);
% for i = 1:9
%     resized_train_alpha(i:i+numtrain-1,:) = train_alpha(i,:,:);
%     resized_test_alpha(i:i+numtest-1,:) = test_alpha(i,:,:);
% end
resized_alpha = zeros(180,128);
resized_beta = zeros(180,128);
resized_theta = zeros(180,128);


for i = 1:9
    %alpha(i,:,:)
    resized_alpha(total*(i-1)+1:total*(i-1)+total,:) = alpha(i,:,:);
    resized_beta(total*(i-1)+1:total*(i-1)+total,:) = beta(i,:,:);
    resized_theta(total*(i-1)+1:total*(i-1)+total,:) = theta(i,:,:);
end


% X = [1,2;3,4;7,1;9,9;5,7;-1,2;-3,4;-7,1;-9,9;-5,7;
%     -1,-2;-3,-4;-7,-1;-9,-9;-5,-7;1,-2;3,-4;7,-1;9,-9;5,-7;]
% Y = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4];
% Z = [-1,8;-6,-4;7,-9;8,3;5,9;-6,-7;3,-2];
% W = [2;3;4;1;1;3;4];
% Mdl = fitcecoc(X,Y);
% Y_predicted = predict(Mdl,Z);
% Accuracy = (nnz((Y_predicted==W))/numel(Y_predicted))*100

% Mdl = fitglm(resized_train_alpha,train_Y);
% Y_predicted = predict(Mdl,resized_test_alpha);
% Y_predicted = cell2mat(Y_predicted);
% Y_predicted = str2num(Y_predicted);
% Accuracy = (nnz((Y_predicted==test_Y))/numel(Y_predicted))*100

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


% alpha_idx = kmeans(resized_alpha,2);
% beta_idx = kmeans(resized_beta,2,'Distance','correlation');
% theta_idx = kmeans(resized_theta,2,'Distance','correlation');


answerclass = zeros(1,180); %row matrix

original_resized_alpha = resized_alpha;
sno = [1:180]';
resized_alpha = [sno,resized_alpha];
alpha_idx1 = kmedoids(resized_alpha,2,'Distance','spearman');
beta_idx1 = kmedoids(resized_beta,2,'Distance','spearman');
theta_idx1 = kmedoids(resized_theta,2,'Distance','spearman');


answerclass = performKmeans(resized_alpha, answerclass)
reshaped_answerclass = reshape(answerclass,20,9);
[mode, freq] = mode(reshaped_answerclass);

% Confusion Matrix

confmat = zeros(9,16);

for i = 1:9
    for j = 1:20
        get = reshaped_answerclass(j,i);
        confmat(i,get) = confmat(i,get)+1; 
    end
end

new_confmat = [confmat(:,11),confmat(:,10),confmat(:,14),confmat(:,9),confmat(:,1),confmat(:,3),confmat(:,2),confmat(:,5),confmat(:,4),confmat(:,6),confmat(:,7),confmat(:,8),confmat(:,12),confmat(:,13),confmat(:,15),confmat(:,16)];


% figure; hist(beta_idx); title('beta_idx');
% figure; hist(beta_idx1); title('beta_idx1');
% 
% figure; hist(alpha_idx); title('alpha_idx');
% figure; hist(alpha_idx1); title('alpha_idx1');
% 
% figure; hist(theta_idx); title('theta_idx');
% figure; hist(theta_idx1); title('theta_idx1');

% temp = find(alpha_idx1==2);
% temp2 = reshape(alpha_idx1,20,9);
% classif = zeros(1,9);
% 
% for i = 1:20
%     classif = classif + (temp2(i,:) == 2);
% end
