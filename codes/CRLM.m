function [mean1,sigma_est,indice]=CRLM(x,k,sigma_max)

t=0;
mean1=[];
[n,d]=size(x);
%  sigma_max=10;
sigma_est=[];

x1=x;
ind{1}=[1:n];
indice=[];

while t<k
[mu,var_est,~]=simplifiedalg(x1,sigma_max);
clear id;
mean3=mu;
sigma1=sqrt(var_est);
mean1=[mean1;mean3];
sigma_est=[sigma_est;sigma1];
% R_sigma_est2=4*d*sigma1^2;
[~,~,id,~]=robust_new_loss(x1,mean3,sigma1);
x1(id,:)=[];
t=t+1;
ind_temp=ind{t};
indice{t}=ind_temp(id);
temporary=ind{t};
temporary(id)=[];
ind{t+1}=temporary;
end
indice{t+1}=ind{t+1};
