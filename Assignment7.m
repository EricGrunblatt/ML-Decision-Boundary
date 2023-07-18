% Assignment 7: Linear Hard-Margin Support Vector Machine
% By: Eric Grunblatt

X = dlmread("X_LinearSeparable.txt");
Y = dlmread("Y_LinearSeparable.txt");
numRows = size(X,1); % d = numRows
numCols = size(X,2); % N = numCols

%%%%%% A. Implement linear support vector machine to obtain (w, b) %%%%%%
H = zeros(numRows+1, numRows+1);
for i=1:size(H,1)
    for j=1:size(H,2)
        if(i == j && i ~= 1)
            H(i,j) = 1;
        end
    end
end
phi = zeros(numRows+1, 1);
A = zeros(numCols, numRows+1);
for i=1:numCols
    temp = -Y(i) * transpose(X(:,i));
    A(i,1) = -Y(i);
    A(i,2) = temp(1);
    A(i,3) = temp(2);
end
c = zeros(numCols, 1);
c(1:end) = -1;

q = quadprog(H, phi, A, c);

%%%%%% B. Identify which training samples are support vectors %%%%%%
figure;
b = q(1); % Identify b
w = q(2:end); % Identify w
for i=1:numCols
    sv = Y(i) * ((transpose(w) * X(:,i)) + b);
    if(round(sv,6) == double(1))
        p = plot(X(1,i), X(2,i), 's', 'color', 'green');
        set(p, 'linewidth',4);
        fprintf('Support Vector Coordinate: (%f, %f)\n', X(1,i), X(2,i));
        hold on
    end
end


%%%%%% C. Compute the largest margin you achieved from the SVM %%%%%%
magnitude = transpose(w) * w;
margin = 1/sqrt(magnitude);
fprintf('Largest Margin: %f\n', margin);


%%%%%% D. Highlight support vectors, plot decision boundary and sv lines %%%%%%
slope = -w(1)/w(2);
x = [min(X(1,:)),max(X(1,:))];
y = (slope*x);
line(x, y, 'color', 'magenta');
hold on
y = (slope*x) + margin;
line(x, y, 'color', 'black');
hold on
y = (slope*x) - margin;
line(x, y, 'color', 'black');
hold on

for i=1:numCols
    if(Y(i) == 1) % Y is shown as 1, blue circle
        plot(X(1,i), X(2,i), 'o', 'color', 'blue');
        hold on
    else
        plot(X(1,i), X(2,i), 'x', 'color', 'red');
        hold on
    end
end
hold off