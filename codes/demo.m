%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a demo to implement CRLM on GMMUB data and make comparison of
% other clustering methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Input parameters, generate datasets
clear;
close all;
D = 100;
d = 2;
n = 500;
p = 0.1 ;
sigma = 1;
model = GMMUB(D,n,d,p);
data = model.data;
x =data;
y = model.label;
%% Plot the dataset
figure(1)
plot(x(y==1,1),x(y==1,2),'r*','MarkerSize',3);
hold on;
plot(x(y==2,1),x(y==2,2),'k.','MarkerSize',5)
h=legend('Positive Points',' Negative Points')
set(h,'FontSize',20,'Location','best');
set(gca,'FontSize',30)
grid on;


%% Obtain the labels for different methods

% CRLM
[~,~,index] = CRLM(x,1,5);
label_crlm = convert_label(index,2);
% Spectral Clustering
[label_sc,~,~]=spectral_clustering(x,2,0);      
% k-means
label_km=kmeans(x,2);

% Hierarchical  Clustering

 Z = linkage(x,'complete','euclidean');
 label_hc = cluster(Z,'maxclust',2);
 

%% plot the clustering results
figure (2)
subplot(2,2,1)
plot(x(label_crlm==1,1),x(label_crlm==1,2),'r*','MarkerSize',3);
hold on;
plot(x(label_crlm==2,1),x(label_crlm==2,2),'k.','MarkerSize',5)
title('CRLM')
subplot(2,2,2)
plot(x(label_km==1,1),x(label_km==1,2),'r*','MarkerSize',3);
hold on;
plot(x(label_km==2,1),x(label_km==2,2),'k.','MarkerSize',5)
title('K-means')
subplot(2,2,3)
plot(x(label_sc==1,1),x(label_sc==1,2),'r*','MarkerSize',3);
hold on;
plot(x(label_sc==2,1),x(label_sc==2,2),'k.','MarkerSize',5)
title('Spectral Clustering')
subplot(2,2,4)
plot(x(label_hc==1,1),x(label_hc==1,2),'r*','MarkerSize',3);
hold on;
plot(x(label_hc==2,1),x(label_hc==2,2),'k.','MarkerSize',5)
title('Hierarchical Clustering')

