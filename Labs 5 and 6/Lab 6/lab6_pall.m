%% lab6_parts
% question for mahalanobis dist...don't I find the points assignment by
% taking the distance? 
close all
% load('spikes.mat'); %loads file: 'spikes.mat' (41568x32)
spikedata=spikes;
[sp,t]=size(spikedata);
% -- normalize each data so that they have 0 mean and unit std
for i=1:t
    spikedata(:,i) = (spikes(:,i)-mean(spikes(:,i)))/std(spikes(:,i)); % why do you average the samples and not the snipets? 
end

[u,w,eval]=pca(spikedata);
X = [w(:,1),w(:,2)];

figure(1)
plot(X(1:4:sp,1),X(1:4:sp,2),'k.','linewidth',1.5) %plot only every 4th point
% I can see about 2 to 3 discernable clusters. set K=3

%% Using the centroids method
K=3;
mu=zeros(K,2);
p_last =0;
for i=1:K
    p = ceil(rand()*sp);
    if p==p_last
        p = ceil(rand()*sp);
    end
    mu(i,:)=[X(p,1),X(p,2)];
    p_last=p;
end
mu_new = zeros(size(mu));
err=10;
loop=0;
while err >= 0.0001
    Dist = zeros(sp,K);
    num_K=linspace(1,K,K)';
    for i=1:K
        Dist(:,i) = sqrt((X(:,1)-mu(i,1)).^2+(X(:,2)-mu(i,2)).^2);
    end
    clust_mat = floor(1./(Dist./min(Dist,[],2)));
    sub_X_1 = clust_mat.*X(:,1);
    sub_X_2 = clust_mat.*X(:,2);
    for i=1:K
        sub = [sub_X_1(:,i) sub_X_2(:,i)];
        sub = sub(any(sub,2),:);
        mu_new(i,:)=sum(sub)/length(sub(:,1));
    end
    error = (mu_new-mu)/mu_new;
    err = max(error(:));
    mu=mu_new;
    if loop == 100000
        break
    end
    loop = loop+1;
end

figure(2)
hold on 
for i=1:K
    sub = [sub_X_1(:,i) sub_X_2(:,i)];
    sub = sub(any(sub,2),:);
    if i==1    
        pcol = 'k.';
    elseif i==2
        pcol = 'b.';
    else
        pcol = 'g.';
    end
    plot(sub(:,1),sub(:,2),pcol)
end
plot(mu(:,1),mu(:,2),'rx','linewidth',2)
hold off

av_spike = mu*u(:,1:2)';
figure(3)
subplot(2,2,1)
hold on
for i=1:K
    if i==1    
        pcol = 'b';
    elseif i==2
        pcol = 'g';
    elseif i==3
        pcol = 'r';
    else
        pcol = 'c';
    end
    plot(time,av_spike(i,:),'linewidth',2)
end
hold off

% The results are not the same every time. It requires an initial guess for
% the amount of clusters in the data. 

%% Using the medoids method
% uch better algorithm altogether. At times it still gives the weird
% results, but overall it performs similarly on mutiple occasions before
% messing up

K=3;
mu=zeros(K,2);
p_last =0;
for i=1:K
    p = ceil(rand()*sp);
    if p==p_last
        p = ceil(rand()*sp);
    end
    mu(i,:)=[X(p,1),X(p,2)];
    p_last=p;
end
mu_new = zeros(size(mu));
err=10;
loop=0;
while err >= 0.0001
    Dist = zeros(sp,K);
    num_K=linspace(1,K,K)';
    for i=1:K
        Dist(:,i) = sqrt((X(:,1)-mu(i,1)).^2+(X(:,2)-mu(i,2)).^2);
    end
    clust_mat = floor(1./(Dist./min(Dist,[],2)));
    sub_X_1 = clust_mat.*X(:,1);
    sub_X_2 = clust_mat.*X(:,2);
    for i=1:K
        sub = [sub_X_1(:,i) sub_X_2(:,i)];
        sub = sub(any(sub,2),:);
        mu_new(i,:)=median(sub);
    end
    error = (mu_new-mu)/mu_new;
    err = max(error(:));
    mu=mu_new;
    if loop == 100000
        break
    end
    loop = loop+1;
end

figure(4)
subplot(2,1,1)
hold on 
for i=1:K
    sub = [sub_X_1(:,i) sub_X_2(:,i)];
    sub = sub(any(sub,2),:);
    if i==1    
        pcol = 'k.';
    elseif i==2
        pcol = 'b.';
    else
        pcol = 'g.';
    end
    plot(sub(:,1),sub(:,2),pcol)
end
plot(mu(:,1),mu(:,2),'rx','linewidth',2)
legend('cluster 1', 'cluster 2', 'cluster 3')
hold off

av_spike = mu*u(:,1:2)';
figure(3)
subplot(2,2,2)
hold on
for i=1:K
    if i==1    
        pcol = 'b';
    elseif i==2
        pcol = 'g';
    elseif i==3
        pcol = 'r';
    else
        pcol = 'c';
    end
    plot(time,av_spike(i,:),'linewidth',2)
end
hold off

%% Mahalanobis distance using the assignments from previous part 
loop=0;
% Find the clusters and the groups at the end 
distM = zeros(sp,K);
while loop <= 3
% use sub_X_1 and 2 from up there and from every other loop
    for i=1:K
        sub = [sub_X_1(:,i) sub_X_2(:,i)];
        sub = sub(any(sub,2),:);
        distM(:,i) = mahal(X,sub);
    end
    clust_mat = floor(1./(distM./min(distM,[],2)));
    sub_X_1 = clust_mat.*X(:,1);
    sub_X_2 = clust_mat.*X(:,2);
%     clust_mat = floor(1./(Dist./min(Dist,[],2))); % define the clusters
%     differently
    for i=1:K
        sub = [sub_X_1(:,i) sub_X_2(:,i)];
        sub = sub(any(sub,2),:);
        mu_new(i,:)=median(sub);
    end
    error = (mu_new-mu)/mu_new;
    err = max(error(:));
    mu=mu_new;
    if loop == 100000
        break
    end
    loop = loop+1;
end
figure(4)
subplot(2,1,2)
hold on
for i=1:K
    sub = [sub_X_1(:,i) sub_X_2(:,i)];
    sub = sub(any(sub,2),:);
    if i==1    
        pcol = 'k.';
    elseif i==2
        pcol = 'b.';
    else
        pcol = 'g.';
    end
    plot(sub(:,1),sub(:,2),pcol)
end
plot(mu(:,1),mu(:,2),'rx','linewidth',2)
legend('cluster 1', 'cluster 2', 'cluster 3')
hold off

av_spike = mu*u(:,1:2)';
figure(3)
subplot(2,2,4)
hold on
for i=1:K
    if i==1    
        pcol = 'b';
    elseif i==2
        pcol = 'g';
    elseif i==3
        pcol = 'r';
    else
        pcol = 'c';
    end
    plot(time,av_spike(i,:),'linewidth',2)
end
hold off

%% Mixture of Gaussians
K=3;
options = statset('display','final');
obj = gmdistribution.fit(X,K,'options',options);
clusters = cluster(obj,X);
centroids = obj.mu;
figure(5)
hold on
for i=1:4:sp
    if clusters(i)==1    
        pcol = 'b.';
    elseif clusters(i)==2
        pcol = 'g.';
    elseif clusters(i)==3
        pcol = 'r.';
    else
        pcol = 'c.';
    end
    plot(X(i,1),X(i,2),pcol)
end
plot(centroids(:,1),centroids(:,2),'kx','linewidth',2.5)
figure(6)
h = ezcontour(@(x,y)pdf(obj,[x,y]),[-15 15],[-15 15]);

av_spike = centroids*u(:,1:2)';
figure(3)
subplot(2,2,3)
hold on
for i=1:K
    if i==1    
        pcol = 'b';
    elseif i==2
        pcol = 'g';
    elseif i==3
        pcol = 'r';
    else
        pcol = 'c';
    end
        plot(time,av_spike(i,:),pcol,'linewidth',2)
end
hold off

%% Own algorithm 
