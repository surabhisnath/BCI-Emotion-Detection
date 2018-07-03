function [answerclass,sizes] = performKmeans(resized_alpha, answerclass, sizes)
    if size(resized_alpha,1) <= 20
        t = getGlobalx();    % get counter
        sizes(1,t) = size(resized_alpha,1);
        answerclass(resized_alpha(:,1)) = t;
        setGlobalx(t+1);
        return;
    else
        alpha_idx1 = kmedoids(resized_alpha(:,2:129),2);
        resized_alpha1 = resized_alpha(find(alpha_idx1==1),:);
        resized_alpha2 = resized_alpha(find(alpha_idx1==2),:);
    
        [answerclass,sizes] = performKmeans(resized_alpha1, answerclass, sizes);
        [answerclass,sizes] = performKmeans(resized_alpha2, answerclass, sizes);
    end
end

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