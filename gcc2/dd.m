function [r rx]=dd(x)

ll=length(x);
rx=[];
rxx = zeros(1, 4);

k0=1;
for k=1:ll
  if k==ll || x(k+1,2) != x(k0, 2)
    rk=polyfit(x(k0:k,3), x(k0:k,4), 1);
    rxx(1:2)=x(k,1:2);
    rxx(3:4)=rk;
    rx=[rx; rxx];
    k0 = k + 1;
  end
end

mm=zeros(4);
b=zeros(4,2);
lk=length(rx);

mm(1,1)=lk;
mm(1,2)=sum(rx(:,1));                # m
mm(1,3)=sum(rx(:,2));                # n
mm(1,4)=sum(rx(:,1).*rx(:,2));       # m*n
b (1,1)=sum(rx(:,3));                # y0
b (1,2)=sum(rx(:,4));                # y1

mm(2,1)=mm(1,2);
mm(2,2)=sum(rx(:,1).*rx(:,1));                 # m
mm(2,3)=sum(rx(:,1).*rx(:,2));                 # n
mm(2,4)=sum(rx(:,1).*rx(:,1).*rx(:,2));        # m*n
b (2,1)=sum(rx(:,1).*rx(:,3));                 # y0
b (2,2)=sum(rx(:,1).*rx(:,4));                 # y1

mm(3,1:2)=mm(1:2,3);
mm(3,3)=sum(rx(:,2).*rx(:,2));                 # n
mm(3,4)=sum(rx(:,2).*rx(:,1).*rx(:,2));        # m*n
b (3,1)=sum(rx(:,2).*rx(:,3));                 # y0
b (3,2)=sum(rx(:,2).*rx(:,4));                 # y1

mm(4,1:3)=mm(1:3,4);
mm(4,4)=sum(rx(:,1).*rx(:,2).*rx(:,1).*rx(:,2)); # m*n
b (4,1)=sum(rx(:,1).*rx(:,2).*rx(:,3));          # y0
b (4,2)=sum(rx(:,1).*rx(:,2).*rx(:,4));          # y1

mt=chol(mm);
r1=mt\(mt'\b(:,1));
r0=mt\(mt'\b(:,2));

r(1)=r0(1); # const
r(2)=r0(2); # m
r(3)=r0(3); # n
r(4)=r1(1); # k
r(5)=r0(4); # m*n
r(6)=r1(2); # k*m
r(7)=r1(3); # k*n
r(8)=r1(4); # m*n*k

r=r(:);
