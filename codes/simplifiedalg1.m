% function [mu,var_est,ind]=simplifiedalg1(x1,sigma_max)
% %selection of h?
% %some adaptation to make it to be used for multiple positive clusters
% [n1,d]=size(x1);
%  loss=zeros(1,n1);
% 
% % R_sigma=2*sigma(i)^2*(log(V)-d*log(sigma(i)));
% R_sigma2=4*d*sigma_max^2;
% distance1=pdist2(x1,x1);
% 
% for j=1:n1
%  
%  [loss(j),~,~,~]=robust_new_loss(x1,x1(j,:),sigma_max);
% end
% [~,m2]=min(loss);
% dis=distance1(m2,:);
% ind=find(dis<sqrt(R_sigma2));
%  if length(ind)==1
%      mu=x1(m2,:);
%      var_est=sigma_max^2;
%  else
%  mu=mean(x1(ind,:));
% 
% 
%  var_est=mean(var(x1(ind,:)));
% 
%  end

%putting a sqrt is right in theory but in practical it is  not good 
%[loss(j),~,~,~]=robust_new_loss(x,mu,sqrt(var1(j)))



function [mu,var_est,ind]=simplifiedalg1(x1,sigma_max,distance1,F)
%selection of h?
%some adaptation to make it to be used for multiple positive clusters
[n1,d]=size(x1);
 loss=zeros(1,n1);

% R_sigma=2*sigma(i)^2*(log(V)-d*log(sigma(i)));

%R_sigma2=4*d*sigma_max^2;
%F=16;
R_sigma2=F*d*sigma_max^2;
%distance1=pdist2(x1,x1);

for j=1:n1
 
 [loss(j),~,~,~]=robust_new_loss(distance1(j,:),d,sigma_max,F);
end
[~,m2]=min(loss);
dis=distance1(m2,:);
ind=find(dis<sqrt(R_sigma2));
 if length(ind)==1
     mu=x1(m2,:);
     var_est=sigma_max^2;
 else
 mu=mean(x1(ind,:));


 var_est=mean(var(x1(ind,:)));

 end