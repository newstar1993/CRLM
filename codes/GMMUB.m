function model=GMMUB(D,n,d,p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Here we construct data of GMM with uniform backgrounds
%n: Number of data points.
%D: radius of the data points
%d: Dimensions
%P: 1xd vector  proportion (0-1) for the positives
%OUTPUTS:
% model: the model parameters containing all the parameters for GMMUB
% data: simulated data
% label: data labels
% true_means: true_means for the positives
% true_sigma: true sigma for isotropic Gaussian of each positive clusters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
y=zeros(n,1);
x=zeros(n,d);
k=length(p);
p=[p;1-sum(p)];
np=mnrnd(n,p);
% forming negatives
mu_n=zeros(1,d);
sigma_n=eye(d);
n_n=mvnrnd(mu_n,sigma_n,np(end));
    x_n=n_n./sqrt(sum(n_n.^2,2));
    x_n=D*(repmat(rand(np(end),1),1,d).^(1/d).*x_n);
% forming positives
A=-(2*D/3)*ones(1,d);
B=(2*D/3)*ones(1,d);
%true_means=(2*D/3)*(2*rand(k,d)-2);
true_sigma=(D/10)*rand(k,1);
x=[];
index=[];
for i =1:k
    % random means
    clear tp;
    true_means(i,:)=unifrnd(A,B);
    tp=mvnrnd(true_means(i,:),true_sigma(i)*eye(d),np(i));
    x=[x;tp];
    index=[index;ones(np(i),1)*i];
end
% adding negatives
x=[x;x_n];
index=[index; ones(np(end),1)*(k+1)];
combined=[x,index];
combined=combined(randperm(size(combined, 1)), :);
x=combined(:,1:d);
index=combined(:,end);
model.data=x;
model.label=index;
model.true_means=true_means;
model.true_sigma=true_sigma;
model.numbers= np;

