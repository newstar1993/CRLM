%% Load data from shape dataset
clear;
%training_positive=imageSet('D:\dataset\shape dataset\1070db\pgms','recursive');
training_positive=imageSet('D:\dropbox\Dropbox\215db','recursive');
N=training_positive(1).Count;
for i=1:N
clear resize;
c=read(training_positive(1),i);
resize=c;
col_n(i)=size(resize,2);
row_n(i)=size(resize,1);
 resize=make_square_size(resize);

 resize=imresize(resize,[256,256]);
  resize(resize<128)=0;
  resize(resize>=128)=255;
  resize=im2double(resize);
 pic(i,:)=reshape(resize,[1,256*256]);
end
% get the boundary points of B: B = bwboundaries(BW)
% [B,L] = bwboundaries(c,'noholes');
% imshow(label2rgb(L, @jet, [.5 .5 .5]))
% hold on
% for k = 1:length(B)
%    boundary = B{k};
%    plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
% end
%% run the PCA to get features
[coeff,score,latent] = pca(pic) ;

%% perform alg2 and k-means on data after PCA

               
 addpath('..\matlab\tensorlab')
 [mu_td,sigma_td, pi_td] = tensorcomp_mult(score',18,0.01);
 mu_td = real(mu_td);

[~,label_td] = min(pdist2(score,mu_td'),[],2);
 
 sigma_max=640;
 [mean_sim,sigma_sim,index]=alg2(score,17,sigma_max);
id_alg2=convert_label(index,18)
 [IDX,C,~]=spectral_clustering(score,18,0);
IDY=kmeans(score,18);
  Z = linkage(x,'complete','euclidean');
  c = cluster(Z,'maxclust',3);
 [IDD, isnoise] = DBSCAN(x,epsilon,2);
 
 %spectral clustering & Hierarchical Clustering
 [IDX,C,~]=spectral_clustering(score,18,0);
%show the image for the first class
%get a subplot to show all the plots for a certain display:
t=4;
k=10;
display_top_images(training_positive,index,k,t);
display_top_images_onerow(training_positive,index,10)
pos_sample=[323,149,41,698,11,24];
display_top_images_onerow(training_positive,pos_sample,1)
display_top_images_onerow(training_positive,pos_sample)
% [acc,previous]=Kimia216_acc(IDY)
% [acc,previous]=Kimia216_acc(IDY)
%% Making Comparison between several clustering methods on accuracy
 label_true=[ones(12,1);ones(12,1).*2;ones(12,1).*3;ones(12,1).*4;ones(12,1).*5;ones(12,1).*6;ones(12,1).*7;ones(12,1).*8
     ones(12,1).*9;ones(12,1).*10;ones(12,1).*11;ones(12,1).*12;ones(12,1).*13;ones(12,1).*14;ones(12,1).*15;ones(12,1).*16;ones(12,1).*17;ones(12,1).*18];

%%
label_td(label_td==13)=1;
label_td(label_td==8)=2;
label_td(label_td==11)=3;
label_td(label_td==14)=4;
label_td(label_td==15)=5;
label_td(label_td==16)=6;
label_td(label_td==17)=7;
[Acc_td,~]=get_acc(label_true,label_td);
 %  sigma_max=100:100:1500;
%  k=18;
%  j=length(sigma_max);
%  for i=1:j
%      [mean_sim,sigma_sim,index]=alg2(score,k-1,sigma_max(i));
%      id_alg2=convert_label(index,k);
%      [Acc_alg2(i),rand_index,match]=AccMeasure(label_true,id_alg2);
%  end
%  plot(sigma_max,Acc_alg2);
n_runs=20;
for i=1:n_runs
IDY=kmeans(score,18);
[Acc_kmeans(i),previous]=get_acc(label_true,IDY);
end
Acc_kmean=mean(Acc_kmeans);
% fprintf('clustering accracy of kmeans: %f',Acc_kmeans)
% fprintf('\n')
  Z = linkage(score,'complete','euclidean');
  c = cluster(Z,'maxclust',18);
  [Acc_CL,~]=get_acc(label_true,c);
fprintf('clustering accracy of Complete Link: %f',Acc_CL)
fprintf('\n')
epsilon=100:100:1000;
p=2;
for i=1:length(epsilon)
    for j=1:length(p)
 [IDD, isnoise] = DBSCAN(score,epsilon(i),p(j));
  [Acc_DBSCAN(i,j),~]= get_acc(label_true,IDD);
    end
end
fprintf('clustering accracy of DBSCAN: %f',Acc_DBSCAN)
fprintf('\n')
[y1, model, L] = mixGaussVb(score',20);
[z1,model,llh] = mixGaussEm(score',18); 
  [Acc_EM,~]=get_acc(label_true,z1');
  fprintf('clustering accracy of EM: %f',Acc_EM)
fprintf('\n')
sigma_max=640;
% sigma_max=50:10:1500;
n_sig=size(sigma_max,2);
for i=1:n_sig
    [mean_sim,sigma_sim,index]=alg2(score,17,sigma_max(i));
    id_alg2=convert_label(index,18);
    [Acc_alg2(i),~]=get_acc(label_true,id_alg2);
    fprintf('clustering accracy of our algorithms with sigma_max=%f : %f',sigma_max(i) ,Acc_alg2(i))
    fprintf('\n')
end
logV=sum(log(range(score)));
ID_em  = mixGauss_uniform_backEm(score',18,logV);
[Acc_em,~]=get_acc(label_true,ID_em');

save comparison_clustering_Kimia_216 Acc_kmeans Acc_DBSCAN Acc_CL Acc_alg2 score 
%% Modeling clustering data for 1070 shape database
% construct true_labels
clear;
training_positive=imageSet('D:\dropbox\Dropbox\pgms','recursive');
N=training_positive(1).Count;
label=zeros(N,1);
label(1:20)=1;
label(21:40)=2;
label(41:60)=3;
label(61:86)=4;
label(87:91)=5;
label(92:141)=6;
label(142)=7;
label(143:147)=8;
label(148:180)=9;
label(181:200)=10;
label(201:204)=11;
label(205:206)=12;
label(207:208)=13;
label(209:220)=14;
label(221:232)=15;
label(233:252)=16;
label(253:301)=17;
label(302:321)=18;
label(322:341)=19;
label(342:361)=20;
label(362:373)=21;
label(374:395)=22;
label(396:411)=23;
label(412)=24;
label(413:414)=25;
label(415:429)=26;
label(430:478)=27;
label(479:480)=28;
label(481)=29;
label(482:505)=30;
label(506:507)=31;
label(508:534)=32;
label(535:540)=33;
label(541:560)=34;
label(561:580)=35;
label(581:620)=36;
label(621:652)=37;
label(653:669)=38;
label(670:694)=39;
label(695:714)=40;
label(715:746)=41;
label(747:762)=42;
label(763:768)=43;
label(769:771)=44;
label(772:791)=45;
label(792)=46;
label(793:812)=47;
label(813:828)=48;
label(829:830)=49;
label(831:842)=50;
label(843:846)=51;
label(847:851)=52;
label(852:854)=53;
label(855:858)=54;
label(859:860)=55;
label(861:880)=56;
label(881:890)=57;
label(891:915)=58;
label(916)=59;
label(917:920)=60;
label(941:944)=61;
label(945:947)=62;
label(948:1007)=63;
label(1008:1048)=64;
label(1049:1070)=65;
label(921:940)=66;
%% preprocessing
label_true=label;
for i=1:N
clear resize;
c=read(training_positive(1),i);
resize=c;
col_n(i)=size(resize,2);
row_n(i)=size(resize,1);
 resize=make_square_size(resize);

 resize=imresize(resize,[256,256]);
  resize(resize<128)=0;
  resize(resize>=128)=255;
  resize=im2double(resize);
 pic(i,:)=reshape(resize,[1,256*256]);
end
[coeff,score,latent] = pca(pic) ;
dim_td =200;
reducedDimension = coeff(:,1:200);
reducedData = pic * reducedDimension;
score = reducedData;
sigma_max=240;
% sigma_max=50:10:1500;
n_sig=size(sigma_max,2);
    [mean_sim,sigma_sim,index]=alg2(score,15,sigma_max(i));
    id_alg2=convert_label(index,15);

t=4;
k=8;
display_top_images(training_positive,index,k,t)
%% Constructing Positive and negatives and then take the test
% selecting heart,tome and 
indice=[1,2,3,19,40,9];
indice_show=[2,23,45,150,700,323];
% for i=6
%     fig = figure;
%     imshow(read(training_positive(1),indice_show(i)));
%     axis off;
%     print(fig,'pos_example_6','-dpng')
% end
ind_label=[1:66];
k=size(indice,2);
k_p=10;
neg_label=label;
n_p=120;
for i=1:size(indice,2)
    clear ind;
    ind=find(label==indice(i));
%     new_label(ind,:)=i;
    ind_label(ind_label==indice(i))=[];
    neg_label(neg_label==indice(i))=[];
end
neg_score=score((ismember(label,ind_label)==1),:);
orig_score=[];
new_index=[];
%Positives
for i=1:k
        clear indice_tempo
        indice_tempo=find(label==indice(i));     
        orig_score(((i-1)*k_p+1):(i*k_p),:)=score(indice_tempo(1:k_p),:);
        new_index(((i-1)*k_p+1):(i*k_p),:)=i;
end
% Negatives
for i=1:n_p
    if  i<=size(ind_label,2)
        indice_temp=find(neg_label==ind_label(i));
        c=size(indice_temp,1);
        m=floor(rand(1)*c)+1;
        indice_temp1=indice_temp(m,:);
        indice_temp(m,:)=[];
        orig_score(k_p*k+i,:)=neg_score(indice_temp1,:);
        neg_score(indice_temp1,:)=[];
        neg_label(indice_temp1)=[];
    else
        temp_size=size(neg_score,1);
       p= randperm(temp_size);
        orig_score(k_p*k+i,:)=neg_score(p(1),:);
        neg_score(p(1),:)=[];
    end
    new_index(k_p*k+i,:)=k+1;
end
%% comparison
label_true=new_index;
score=orig_score;
n_runs=20;
dim_new =200
score = score;

addpath('..\matlab\tensorlab')
 [mu_td,sigma_td, pi_td] = tensorcomp_mult(score',7,0.01);
 mu_td = real(mu_td);
 [~,label_td] = min(pdist2(score,mu_td'),[],2);
  [~,pre_td,rc_td]=eva_cluster_multiple(label_true,label_td,1);
 
for i=1:n_runs
IDY=kmeans(score,7);
[Acc_kmeans(i)]=AccMeasure(label_true,IDY);
[purity(i),f(i),~,pre_km(i),rc_km(i)]=eva_cluster_multiple(label_true,IDY,1);
end

save  clustering_data orig_socre

% Spectral Clustering
% sigma=1;
%  W = SimGraph_Full(score', sigma);
% [C, L, U] = SpectralClustering(W, 7, 2);
% [clusters, evalues, evectors] = spcl(score', 7);
% [IDX,C,~]=spectral_clustering(orig_score,7,0);
% affinity = CalculateAffinity(orig_score);
affinity = CalculateAffinity(score);
for i=1:size(affinity,1)
    D(i,i) = sum(affinity(i,:));
end
for i=1:size(affinity,1)
    for j=1:size(affinity,2)
        NL1(i,j) = affinity(i,j) / (sqrt(D(i,i)) * sqrt(D(j,j)));  
    end
end

% compute the normalized laplacian (method 2)  eye command is used to
% obtain the identity matrix of size m x n
% NL2 = eye(size(affinity,1),size(affinity,2)) - (D^(-1/2) .* affinity .* D^(-1/2));

% perform the eigen value decomposition
[eigVectors,eigValues] = eig(NL1);

% select k largest eigen vectors
k = 7;
nEigVec = eigVectors(:,(size(eigVectors,1)-(k-1)): size(eigVectors,1));

% construct the normalized matrix U from the obtained eigen vectors
for i=1:size(nEigVec,1)
    n = sqrt(sum(nEigVec(i,:).^2));    
    U(i,:) = nEigVec(i,:) ./ n; 
end

%% TD
 addpath('..\matlab\tensorlab')
 [mu_td,sigma_td, pi_td] = tensorcomp_mult(score',7,0.01);
 mu_td = real(mu_td);
 [~,label_td] = min(pdist2(score,mu_td'),[],2);
  [~,pre_,rc_cl]=eva_cluster_multiple(label_true,IDX,1);
 %%
% perform kmeans clustering on the matrix U
[IDX,C] = kmeans(U,k); 
[purity_SC,f_SC,RI_SC,pre_km_SC,rc_km_SC]=eva_cluster_multiple(label_true,IDX,1);
% K-means

Acc_kmean=mean(Acc_kmeans);
% fprintf('clustering accracy of kmeans: %f',Acc_kmeans)
% fprintf('\n')
  Z = linkage(score,'complete','euclidean');
  c = cluster(Z,'maxclust',7);
  [Acc_CL]=AccMeasure(label_true,c);
  [~,~,~,pre_cl,rc_cl]=eva_cluster_multiple(label_true,c,2);
fprintf('clustering accracy of Complete Link: %f',Acc_CL)
fprintf('\n')
epsilon=22000;

% Spectral Clustering
[IDX,C,~]=spectral_clustering(score,7,0);
[Acc_SC]=AccMeasure(label_true,c);
 [~,~,~,pre_cl,rc_cl]=eva_cluster_multiple(label_true,IDX,1);
  
p=2;
clear Acc_DBSCAN
for i=1:length(epsilon)
    for j=1:length(p)
 [IDD, isnoise] = DBSCAN(score,epsilon(i),p(j));
 [acc,ind_reassign]=get_acc(label_true,IDD);
   m=max(ind_reassign);
  ind_reassign(ind_reassign==0)=m+1;
  [Acc_DBSCAN(i,j)]= AccMeasure(label_true,ind_reassign);
  [~,~,~,pre_db,rc_db]=eva_cluster_multiple(label_true,ind_reassign,2);
    end
end
[y1, model, L] = mixGaussVb(orig_score',7);
[z1,model,llh] = mixGaussEm(orig_score',7); 
%sigma_max=380;
sigma_max=380;
n_sig=length(sigma_max);
for i=1:n_sig
    [mean_sim,sigma_sim,index]=alg2(score,6,sigma_max(i));
    id_alg2=convert_label(index,7);
    [Acc_alg2(i)]=AccMeasure(label_true,id_alg2);
    [~,~,~,pre_alg2(i),rc_alg2(i)]=eva_cluster_multiple(label_true,id_alg2,2);
    fprintf('clustering accracy of our algorithms with sigma_max=%f : %f',sigma_max(i) ,Acc_alg2(i))
    fprintf('\n')
end
logV=sum(log(range(score)));
ID_em  = mixGauss_uniform_backEm(score',7,logV);
[Acc_em,ind_reassign]=get_acc(label_true,ID_em');
[~,~,~,pre_em,rc_em]=eva_cluster_multiple(label_true,ind_reassign,2);
%save shape_1070db_6positives_neg120_new
