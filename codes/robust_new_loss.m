function [loss,nn,id,mu]=robust_new_loss(x,mu,sigma)
[n,d]=size(x);
% mx=min(x);
% Mx=max(x);
% dx=Mx-mx;
% logV=sum(log(dx));
lossx=sum(((x-repmat(mu,[n,1])).^2)/(d*sigma^2),2)-4;
lossx=min(lossx,zeros(n,1));
loss=sum(lossx);
nn=sum(lossx<0);
id=find(lossx<0);
if n>0
    mu=loss/n;
else
    mu=0;
end