function [Pv,obj]=CCSFS(fea,lambda,alpha,beta,n,v,c)

eta=1e8;
[v1,v2]=size(fea);
Pv=cell(v1,v2);
Gv=cell(v1,v2);
Dv=cell(v1,v2);
DWv=cell(v1,v2);
Kv=cell(v1,v2);
Wv=zeros(v2,1);
d=zeros(v2,1);
Y=randn(n,c);
Ivector=ones(n,1);
MaxIter=30;

for num = 1:v
    fea{num}=fea{num}';
    d(num)=size(fea{num},1);
    Kv{num}=fea{num}'*fea{num};
    Gv{num}=randn(n,c);
    DWv{num}=eye(d(num),d(num));
    Wv(num)=1/v;
end

for iter=1:MaxIter
%     disp(iter);
    for num = 1:v
        Pv{num}=(alpha*fea{num}*fea{num}'+beta*DWv{num})\(alpha*fea{num}*Y);
        Pi=sqrt(sum(Pv{num}.*Pv{num},2)+eps);
        diagonal=0.5./Pi;
        DWv{num}=diag(diagonal);
        GG=Gv{num}*Gv{num}';
        Ggradient=((2*Kv{num}*Gv{num}+2*lambda*Wv(num)*Y*Y'*Gv{num}+2*eta*Ivector*Ivector'*Gv{num})./(GG*Kv{num}*Gv{num}+Kv{num}*GG*Gv{num}+2*lambda*Wv(num)*GG*Gv{num}+eta*Ivector*Ivector'*GG*Gv{num}+eta*GG*Ivector*Ivector'*Gv{num})).^(1/4);
        Gv{num}=Gv{num}.*Ggradient;
    end

    sumWv=0;
    sumWvGG=0;
    sumXP=0;
    for num = 1:v
        sumWv=sumWv+Wv(num);
        sumWvGG=sumWvGG+Wv(num)*Gv{num}*Gv{num}';
        sumXP=sumXP+fea{num}'*Pv{num};
    end
    Ygradient=((2*lambda*sumWvGG*Y+alpha*sumXP)./(2*lambda*sumWv*Y*Y'*Y+alpha*v*Y)).^(1/4);
    Y=Y.*Ygradient;
    for num = 1:v
        Wv(num)=1/(2*norm(Y*Y'-Gv{num}*Gv{num}','fro'));
    end
    
    sumobj=0;
    for num = 1:v
       sumobj=sumobj+real(norm(fea{num}-fea{num}*Gv{num}*Gv{num}','fro').^2+lambda*Wv(num)*norm(Y*Y'-Gv{num}*Gv{num}','fro').^2+alpha*norm(fea{num}'*Pv{num}-Y,'fro').^2+beta*trace(Pv{num}'*DWv{num}*Pv{num}); 
    end
    obj(iter)=sumobj;
    if iter >= 2 && (abs(obj(iter)-obj(iter-1)/obj(iter))<eps)
        break;
    end
    
end

end

