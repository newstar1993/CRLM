function [IDX,C,U]=spectral_clustering(X,n,type)
%X :data matrix n:number of clusters
%n: number of clusters you want to make
affinity = CalculateAffinity(X);
if type == 0
end
if type==1
  affinity_A=affinity;
%figure,imshow(affinity,[]), title('Affinity Matrix')
% compute the degree matrix
%compute affinity_A
sa=sort(affinity_A(:),'descend');
thr=sa(size(affinity_A,1)*2);
affinity_A(affinity_A<thr)=0;  
affinity=affinity_A;
end
if type==2
affinity_B=affinity;
[nrow,~]=size(affinity_B);
for i=1:nrow
    sa1=sort(affinity_B(i,:),'descend');
    thr1=sa1(2);
    affinity_c=affinity_B(i,:);
    affinity_c(affinity_c<thr1)=0;
    affinity_B(i,:)=affinity_c;
end
affinity=affinity_B;
end
for i=1:size(affinity,1)
    D(i,i) = sum(affinity(i,:));
end
for i=1:size(affinity,1)
    for j=1:size(affinity,2)
        NL1(i,j) = affinity(i,j) / (sqrt(D(i,i)) * sqrt(D(j,j)));  
    end
end
[eigVectors,eigValues] = eig(NL1);
k =n;
%nEigVec = eigVectors(:,(size(eigVectors,1)-(k-1)): size(eigVectors,1));
nEigVec = eigVectors(:,1:k);
for i=1:size(nEigVec,1)
    n1 = sqrt(sum(nEigVec(i,:).^2));    
    U(i,:) = nEigVec(i,:) ./ n1; 
end
[IDX,C] = kmeans(U,k); 

% affinity_A=affinity;
% %figure,imshow(affinity,[]), title('Affinity Matrix')
% % compute the degree matrix
% %compute affinity_A
% sa=sort(affinity_A(:),'descend');
% thr=sa(size(affinity_A,1)*2);
% affinity_A(affinity_A<thr)=0;
% affinity_B=affinity;
% [nrow,~]=size(affinity_B);
% for i=1:nrow
%     sa1=sort(affinity_B(i,:),'descend');
%     thr1=sa1(n*2);
%     affinity_c=affinity_B(i,:);
%     affinity_c(affinity_c<thr1)=0;
%     affinity_B(i,:)=affinity_c;
% end
% affinity=max(affinity_A,affinity_B);
% for i=1:size(affinity,1)
%     D(i,i) = sum(affinity(i,:));
% end
% for i=1:size(affinity,1)
%     for j=1:size(affinity,2)
%         NL1(i,j) = affinity(i,j) / (sqrt(D(i,i)) * sqrt(D(j,j)));  
%     end
% end
% [eigVectors,eigValues] = eig(NL1);
% k =n;
% %nEigVec = eigVectors(:,(size(eigVectors,1)-(k-1)): size(eigVectors,1));
% nEigVec = eigVectors(:,1:k);
% for i=1:size(nEigVec,1)
%     n1 = sqrt(sum(nEigVec(i,:).^2));    
%     U(i,:) = nEigVec(i,:) ./ n1; 
% end
% [IDX,C] = kmeans(U,k); 

%IN case we need the means
% mean1=zeros(k,1);
%  for i =1:k
%      mean1(i)=mean(X(IDX==i));
%  end
