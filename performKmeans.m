function answerclass = performKmeans(resized_alpha, answerclass)
    if size(resized_alpha,1) <= 20
        disp('hi')
        size(resized_alpha,1)
        t = getGlobalx()
        answerclass(resized_alpha(:,1)) = t;
        setGlobalx(t+1);
        return;
    else
        disp('hello')
        alpha_idx1 = kmedoids(resized_alpha,2);
        resized_alpha1 = resized_alpha(find(alpha_idx1==1),:);
        resized_alpha2 = resized_alpha(find(alpha_idx1==2),:);
    
        answerclass = performKmeans(resized_alpha1, answerclass);
        answerclass = performKmeans(resized_alpha2, answerclass);
    end
end