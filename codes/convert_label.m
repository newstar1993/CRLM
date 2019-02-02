function label=convert_label(indices,n)
%% this function is used to convert clustering groups in indices to become labels.
label=zeros(n,1);
k=size(indices,2);
for i=1:k
    label(indices{i},:)=i;
end
