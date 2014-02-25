clear all;

x = linspace(-5, 5);
y = linspace(-5, 5);
z = zeros(100,100);

%part ii
%{
mu1 = [-1 2];
sigma1 = [3 1; 1 2];
%}

%part iii
%{
mu1 = [0 2];
mu2 = [2 0];
sigma1 = [1 1; 1 2];
sigma2 = [1 1; 1 2];
%}

%part iv
%{
mu1 = [0 2];
mu2 = [2 0];
sigma1 = [1 1; 1 2];
sigma2 = [3 1; 1 2];
%}

%part v
mu1 = [1 1];
mu2 = [-1 -1];
sigma1 = [1 0; 0 2];
sigma2 = [2 1; 1 2];
for i = 1:100
    for j = 1:100
        z(i,j) = mvnpdf([x(i), y(j)], mu1, sigma1) - mvnpdf([x(i), y(j)], mu2, sigma2);
    end
end

contour(x, y, z);