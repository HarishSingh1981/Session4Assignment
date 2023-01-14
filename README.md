# Session4Assignment
It contains the solution of assignments in SchoolOfAI AI course

PART1:

![image](https://user-images.githubusercontent.com/13793213/212476994-b8112176-ba47-4d4d-8141-58f902c9d97f.png)


1. Attached excel sheet depicts the back-propogation with simple neural network with 1 input/1 hidden and 1 output layer.
C1 and C2 --> input neurons
h1 and h1 --> hidden layer neurons
out_h1 and out_h2 are output of sigmoid activation function on h1 and h2 respectively
o1 and o2 are output neurons where out_o1 and out_o2 are result of sigmoid activation on o1 and o2 respectively.
Weights as depicted in figure w1/w2/w3/w4/w5/w6/w7/w8 are randomly initialized with given number on edges.

Below are the major steps involved in back-propogation:
1. Forward pass
Initially network learn with given default weight and produce the 2 output E1 and E2 which summed up as final output E. The
function to calculate output at node h1 and he can be defined as
h1 = w1*c1 + c2*w3    -------- 1
h2 = c1*w2 + c2*w4    -------- 2
out_h1 = 1/1+exp(-h1) [out_h1 is sigmoid function of h1] ------ 3
out_h2 = 1/1+exp(-h2) [out_h2 is sigmoid function of h2] ------ 4

Similarly the final output in forward pass can be calculated as
o1 = out_h1*w5 + out_h2*w7  -------- 5
o2 = out_h1*w6 + out_h2*w8  -------- 6
out_o1 = 1/(1+exp(-o1)) ------------ 7
out_o2 = 1/(1+exp(-o2)) ------------ 8

Subsequent result calculation will be as below:
E1 = 1/2 * (t1 - out_o1)^2 -------- 9
E2 = 1/2 * (t2 - out_o2)^3 -------- 10
E = E1 + E2 ----------------------- 11

Here L2 loss function is taken to calculate loss. t1 and t2 are respective expected output.

2. Back-propogation
The first major step in back-propogation is to calculate the gradient wrt each parameter. Here we have 8 weights as parameter.
So we need to calculate
dE/dw1,dE/dw2,dE/dw3,dE/dw4,dE/dw5,dE/dw6,dE/dw7,dE/dw8

let's start with gradient of E wrt w7
--> We need to follow link from E to w7 and apply chain rule as below
dE/dw7 = d(E1+E2)/dw7 = dE1/dw7 + dE2/dw7 [But E2 is not connected with W7 in anyway]
dE/dw7 = dE1/dw7 = dE1/dout_o1 * dout_o1/do1 * do1/dw7 [Applying chain rule as following the graph]
dE/dw7 = (out_o1 - t1) * out_o1 * (1 - out_o1) * out_h2

Here in above calculation the derivation of sigmoid function f(x) is f(x)*(1-f(x)). Rest we can use above given equations
from 1 to 11 to calculate derivatives.
Similarly the derivatives of dE/dw6,dE/dw5,dE/dw8 can be calculated.

Now to calculate the derivation with E wrt w1/w2/w3/w4 we will first calculate derivative of dE/dh1 and dE/dh2 as
dE/dw1 = dE/dh1 * dh1/dw1 [again the same chain rule]

Now to calculate dE/dh1 we need to follow all the paths leading to h1 from E. So we have these paths:
E-->out_o1-->o1-->out_h1-->h1 [along weight w5]
E-->out_o2-->o2-->out_h1-->h1 [along weight w6]
We need to take sum of derivation along both path as a chain rule so it will be:

dE/dh1 = (out_o1-t1)*(out_o1*(1-out_o1))*w5*(out_h1*(1-out_h1)) + (out_o2-t2)*(out_o2*(1-out_o2))*w6*(out_h1*(1-out_h1))

with the symmetry we can also right dE/dh2. Once both are calculated we can now calculate partial derivatives of E wrt w1,
w2,w3,w4.

3. Update the weights now with partial derivatives
Here we have taken learning rate as alpha. Now all the weights must be updated as below
Wx = Wx - alpha * dE/dWx

4. Network again run the forward pass with updated weights
5. This way networks keep learning

Error Curve for different learning rate:

lr=0.2
![image](https://user-images.githubusercontent.com/13793213/212477126-ba141f85-48d0-4797-925f-8c4c97d476c7.png)

lr=0.5
![image](https://user-images.githubusercontent.com/13793213/212477222-d1f5d5a4-b102-4d62-95f9-5749a403d472.png)

lr=0.8
![image](https://user-images.githubusercontent.com/13793213/212477316-24e1e708-b3f1-4e8c-97fc-d2091ff7181a.png)

lr=2.0
![image](https://user-images.githubusercontent.com/13793213/212477367-e4865b5e-c47e-427b-bbf4-cdfe66e84557.png)
