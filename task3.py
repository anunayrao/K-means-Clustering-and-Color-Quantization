print("---------------------------------------------")	
print("Implementing Task : K-means Clustering")
print("---------------------------------------------")	
print("")                                    
print("---------------------------------------------")	
print("---------- K-means Clustering on X ----------")
print("---------------------------------------------")
import numpy as np
import matplotlib.pyplot as plt
import math
np.random.seed(sum([ord(c) for  c in 'anunayra']))


#Saving figure for only 3 iterations
save = 3 
#Number of Iterations
num_of_iterations = 20
initial_centers_x = [6.2,6.6,6.5]
initial_centers_y = [3.2,3.7,3.0]
def euclidean_distance(x,y):
	distance = math.sqrt((x[0]-y[0])**2 + (x[1] - y[1])**2)
	return distance
data = 10	
I = [[6.2,3.2],[6.6,3.7],[6.5,3.0]]
c = ['(6.2,3.2)','(6.6,3.7)','(6.5,3.0)']
colors = ['r','g','b']


X = [[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],
[5.1,3.8],[6.0,3.0]]



num = 3
C = []
C2 = []
for i in range(num):
	C.append([])
	C2.append([])
	C2[i].append(I[i])

itr = num_of_iterations
count = 0
for k in range(itr):
	CV = []	
	for i in range(data):
		mini = []
		for j in range(num):
			d = euclidean_distance(X[i],I[j])
			mini.append(d)
		m = mini.index(min(mini))
		C[m].append(X[i])
		C2[m].append(X[i])
		CV.append(m)
	print("Classification Vector for Iteration:"+str(k+1),CV)
	for i in range(3):
		plt.scatter(I[i][0],I[i][1],color = colors[i], s = 80)
		s = '('
		st = str(round(I[i][0],4))
		st1 = ','
		st2 = str(round(I[i][1],4))
		t = ')'
		plt.annotate(s+st+st1+st2+t,(I[i][0],I[i][1]))
	if(k>0):
		if(k<save):
			imgn = "task3_iter"+str(k)+"_b.jpg"
			plt.savefig(imgn)
			plt.figure()
		
	for i in range(3):
		plt.scatter(I[i][0],I[i][1],color = colors[i], s = 80)
	list = [y for x in C for y in x]
	j = 0
	

	for t in range(num):
		for i in range(len(C2[t])-1):
			plt.scatter(list[i+j][0],list[i+j][1],marker = '^', edgecolor=colors[t],facecolor =colors[t], s=80)
		j +=len(C2[t])-1
	
	if(k<save):
		imgname = "task3_iter"+str(k+1)+"_a.jpg"
		plt.savefig(imgname)	
		plt.figure()
	
	I = []
	for a in range(num):
		e = np.mean(C2[a], axis = 0)
		I.append(e)
	if(count!=(itr-1)):
		C2 = []
		C=[]
		for b in range(num):
			C2.append([])
			C.append([])
			C2[b].append(I[b])	
		count = count + 1
	
print("Cluster 1 (Red):" )	
print(C2[0])
print("Cluster 2 (Green):")
print(C2[1])
print("Cluster 3 (Blue):")
print(C2[2])
print("Images saved...")

print("---------------------------------------------")	
print("---K-means Clustering: Color Quantization----")
print("---------------------------------------------")

	
import cv2
import numpy as np
np.seterr(over='ignore')
from matplotlib import pyplot as plt
import math
np.random.seed(sum([ord(c) for  c in 'anunayra']))

image = cv2.imread("baboon.jpg")

#Number of Colors:

k = [3,5,10,20]
#Number of iterations for  k-means
itr = 10

h, w, _ = image.shape

X = image.reshape((image.shape[0] * image.shape[1], 3))

def euclidean_distance(x,y):
	distance = math.sqrt((x[0]-y[0])**2 + (x[1] - y[1])**2 + (x[2] - y[2])**2)
	return distance

def kmeansselectcolors(clusters,data,iteration,h,w):
	k = h*w
	r = []	
	count = 0
	X = data
	C = []
	C2 = []
	num = clusters 
	itr = iteration
	for i in range(clusters):
		r.append(np.random.randint(0,k))
	for i in range(num):
		C.append([])
		C2.append([])
		C2[i].append(X[r[i]])

	count = 0
	for k in range(itr):	
		for i in range(k):
			mini = []
			for j in range(num):
				d = euclidean_distance(X[i],X[r[j]])
				mini.append(d)
			m = mini.index(min(mini))
			C[m].append(X[i])
			C2[m].append(X[i])


		#list = [y for x in C for y in x]
		j = 0
	
		I = []
		for a in range(num):
			e = np.mean(C2[a], axis = 0)
			I.append(e)
		if(count!=(itr-1)):
			C2 = []
			C=[]
			for b in range(num):
				C2.append([])
				C.append([])
				C2[b].append(I[b])	

			count = count + 1
	return I	

def kmeansfillcolors(data,h,w,I):
	k = h*w
	I = I	


	for i in range(k):	
		mini = []
		for j in range(len(I)):
			d = euclidean_distance(X[i],I[j])
			mini.append(d)
		m = mini.index(min(mini))
		
		X[i][0] = I[m][0]
		X[i][1] = I[m][1]
		X[i][2] = I[m][2]
	return X
for z in range(len(k)):
	image = cv2.imread("baboon.jpg")
	X = image.reshape((image.shape[0] * image.shape[1], 3))
	I = kmeansselectcolors(k[z],X,itr,h,w)
	X = kmeansfillcolors(X,h,w,I)
	X = X.reshape((h, w, 3))
	imgname = "task3_baboon_"+str(k[z])+".jpg"
	cv2.imwrite(imgname,X)
	print(imgname+ "  created...")

print("---------------------------------------------")	
print("Gaussian Mixture Model : on X ")
print("---------------------------------------------")
import numpy as np
from scipy.stats import multivariate_normal

X = [[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],
[5.1,3.8],[6.0,3.0]]

mu1 = [6.2,3.2]
mu2 = [6.6,3.7]
mu3 = [6.5,3.0]

cov1 = [[0.5, 0],[0,0.5]]
cov2 = [[0.5, 0],[0,0.5]]
cov3 = [[0.5, 0],[0,0.5]]

p1 = 1/3
p2 = 1/3
p3 = 1/3
p = [0.33,0.33,0.33]
# E Step	
md = multivariate_normal.pdf(X,mu1,cov1)*p1
me = multivariate_normal.pdf(X,mu2,cov2)*p2
mf = multivariate_normal.pdf(X,mu3,cov3)*p3
d = np.zeros(10)
e = np.zeros(10)
f = np.zeros(10)

for i in range(10):
	d[i] = md[i]/(md[i]+me[i]+mf[i])
	e[i] = me[i]/(md[i]+me[i]+mf[i])
	f[i] = mf[i]/(md[i]+me[i]+mf[i])



c1 = []
c2 = []
c3 = []

for i in range(10):
	if(d[i]>e[i] and d[i]>f[i]):
		c1.append(i)
		#v1.append(d[i])
	if(e[i]>d[i] and e[i]>f[i]):
		c2.append(i)
		#v2.append(e[i])
	if(f[i]>d[i] and f[i]>e[i]):
		c3.append(i)
		#v3.append(f[i])


# M Step

nr1 = [0 ,0 ]
dr1 = 0
for i in c1:
	nr1 += np.dot(d[i],X[i])
	dr1 +=d[i]
r = nr1/dr1

mu1 = r



nr2 = [0 ,0 ]
dr2 = 0
for i in c2:
	nr2 += np.dot(e[i],X[i])
	dr2 +=e[i]

r = nr2/dr2
mu2 = r

nr3 = [0 ,0]
dr3 = 0
for i in c3:
	nr3 += np.dot(f[i],X[i])
	dr3 +=f[i]
r = nr3/dr3
mu3 = r




p1 = dr1/len(c1)
p2 = dr2/len(c2)
p3 = dr3/len(c3)


sig1 = [0, 0]	
for i in c1:
	sig1 += d[i]*(X[i] - mu1)*(np.transpose(X[i]-mu1))
sig1 /=dr1
cov1 = [[sig1[0],0],[0,sig1[1]]]

sig2 = [0, 0]	
for i in c2:
	sig2 += e[i]*(X[i] - mu2)*(np.transpose(X[i]-mu2))
sig2 /=dr2
cov2 = [[sig2[0],0],[0,sig2[1]]]


sig3 = [0, 0]	
for i in c3:
	sig3 += f[i]*(X[i] - mu3)*(np.transpose(X[i]-mu3))
sig3 /=dr3
cov3 = [[sig3[0],0],[0,sig3[1]]]

print("After First Iterations:")
print("Mu1:",mu1)
print("Mu2:",mu2)
print("Mu3:",mu3) 


print("---------------------------------------------")	
print("Gaussian Mixture Model : Old Faithful Dataset")
print("---------------------------------------------")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
s ="""
1       3.600      79
2       1.800      54
3       3.333      74
4       2.283      62
5       4.533      85
6       2.883      55
7       4.700      88
8       3.600      85
9       1.950      51
10      4.350      85
11      1.833      54
12      3.917      84
13      4.200      78
14      1.750      47
15      4.700      83
16      2.167      52
17      1.750      62
18      4.800      84
19      1.600      52
20      4.250      79
21      1.800      51
22      1.750      47
23      3.450      78
24      3.067      69
25      4.533      74
26      3.600      83
27      1.967      55
28      4.083      76
29      3.850      78
30      4.433      79
31      4.300      73
32      4.467      77
33      3.367      66
34      4.033      80
35      3.833      74
36      2.017      52
37      1.867      48
38      4.833      80
39      1.833      59
40      4.783      90
41      4.350      80
42      1.883      58
43      4.567      84
44      1.750      58
45      4.533      73
46      3.317      83
47      3.833      64
48      2.100      53
49      4.633      82
50      2.000      59
51      4.800      75
52      4.716      90
53      1.833      54
54      4.833      80
55      1.733      54
56      4.883      83
57      3.717      71
58      1.667      64
59      4.567      77
60      4.317      81
61      2.233      59
62      4.500      84
63      1.750      48
64      4.800      82
65      1.817      60
66      4.400      92
67      4.167      78
68      4.700      78
69      2.067      65
70      4.700      73
71      4.033      82
72      1.967      56
73      4.500      79
74      4.000      71
75      1.983      62
76      5.067      76
77      2.017      60
78      4.567      78
79      3.883      76
80      3.600      83
81      4.133      75
82      4.333      82
83      4.100      70
84      2.633      65
85      4.067      73
86      4.933      88
87      3.950      76
88      4.517      80
89      2.167      48
90      4.000      86
91      2.200      60
92      4.333      90
93      1.867      50
94      4.817      78
95      1.833      63
96      4.300      72
97      4.667      84
98      3.750      75
99      1.867      51
100     4.900      82
101     2.483      62
102     4.367      88
103     2.100      49
104     4.500      83
105     4.050      81
106     1.867      47
107     4.700      84
108     1.783      52
109     4.850      86
110     3.683      81
111     4.733      75
112     2.300      59
113     4.900      89
114     4.417      79
115     1.700      59
116     4.633      81
117     2.317      50
118     4.600      85
119     1.817      59
120     4.417      87
121     2.617      53
122     4.067      69
123     4.250      77
124     1.967      56
125     4.600      88
126     3.767      81
127     1.917      45
128     4.500      82
129     2.267      55
130     4.650      90
131     1.867      45
132     4.167      83
133     2.800      56
134     4.333      89
135     1.833      46
136     4.383      82
137     1.883      51
138     4.933      86
139     2.033      53
140     3.733      79
141     4.233      81
142     2.233      60
143     4.533      82
144     4.817      77
145     4.333      76
146     1.983      59
147     4.633      80
148     2.017      49
149     5.100      96
150     1.800      53
151     5.033      77
152     4.000      77
153     2.400      65
154     4.600      81
155     3.567      71
156     4.000      70
157     4.500      81
158     4.083      93
159     1.800      53
160     3.967      89
161     2.200      45
162     4.150      86
163     2.000      58
164     3.833      78
165     3.500      66
166     4.583      76
167     2.367      63
168     5.000      88
169     1.933      52
170     4.617      93
171     1.917      49
172     2.083      57
173     4.583      77
174     3.333      68
175     4.167      81
176     4.333      81
177     4.500      73
178     2.417      50
179     4.000      85
180     4.167      74
181     1.883      55
182     4.583      77
183     4.250      83
184     3.767      83
185     2.033      51
186     4.433      78
187     4.083      84
188     1.833      46
189     4.417      83
190     2.183      55
191     4.800      81
192     1.833      57
193     4.800      76
194     4.100      84
195     3.966      77
196     4.233      81
197     3.500      87
198     4.366      77
199     2.250      51
200     4.667      78
201     2.100      60
202     4.350      82
203     4.133      91
204     1.867      53
205     4.600      78
206     1.783      46
207     4.367      77
208     3.850      84
209     1.933      49
210     4.500      83
211     2.383      71
212     4.700      80
213     1.867      49
214     3.833      75
215     3.417      64
216     4.233      76
217     2.400      53
218     4.800      94
219     2.000      55
220     4.150      76
221     1.867      50
222     4.267      82
223     1.750      54
224     4.483      75
225     4.000      78
226     4.117      79
227     4.083      78
228     4.267      78
229     3.917      70
230     4.550      79
231     4.083      70
232     2.417      54
233     4.183      86
234     2.217      50
235     4.450      90
236     1.883      54
237     1.850      54
238     4.283      77
239     3.950      79
240     2.333      64
241     4.150      75
242     2.350      47
243     4.933      86
244     2.900      63
245     4.583      85
246     3.833      82
247     2.083      57
248     4.367      82
249     2.133      67
250     4.350      74
251     2.200      54
252     4.450      83
253     3.567      73
254     4.500      73
255     4.150      88
256     3.817      80
257     3.917      71
258     4.450      83
259     2.000      56
260     4.283      79
261     4.767      78
262     4.533      84
263     1.850      58
264     4.250      83
265     1.983      43
266     2.250      60
267     4.750      75
268     4.117      81
269     2.150      46
270     4.417      90
271     1.817      46
272     4.467      74
"""
X = np.array([float(v) for v in s.split()]).reshape(-1,3)[:,1:]
l, _ = X.shape

mu1 = [4.0,81]
mu2 = [2.0,57]
mu3 = [4.0,71]

cov1 = [[1.30, 13.98],[13.98,184.82]]
cov2 = [[1.30, 13.98],[13.98,184.82]]
cov3 = [[1.30, 13.98],[13.98,184.82]]

p1 = 1/3
p2 = 1/3
p3 = 1/3
p = [0.33,0.33,0.33]

itr = 5

for z in range(itr):
	# E Step	
	md = multivariate_normal.pdf(X,mu1,cov1)*p1
	me = multivariate_normal.pdf(X,mu2,cov2)*p2
	mf = multivariate_normal.pdf(X,mu3,cov3)*p3
	d = np.zeros(l)
	e = np.zeros(l)
	f = np.zeros(l)

	for i in range(l):
		d[i] = md[i]/(md[i]+me[i]+mf[i])
		e[i] = me[i]/(md[i]+me[i]+mf[i])
		f[i] = mf[i]/(md[i]+me[i]+mf[i])



	c1 = []
	c2 = []
	c3 = []

	for i in range(l):
		if(d[i]>e[i] and d[i]>f[i]):
			c1.append(i)
		if(e[i]>d[i] and e[i]>f[i]):
			c2.append(i)
		if(f[i]>d[i] and f[i]>e[i]):
			c3.append(i)


	# M Step

	nr1 = [0 ,0 ]
	dr1 = 0
	for i in c1:
		nr1 += np.dot(d[i],X[i])
		dr1 +=d[i]
	r = nr1/dr1

	mu1 = r



	nr2 = [0 ,0 ]
	dr2 = 0
	for i in c2:
		nr2 += np.dot(e[i],X[i])
		dr2 +=e[i]

	r = nr2/dr2
	mu2 = r

	nr3 = [0 ,0]
	dr3 = 0
	for i in c3:
		nr3 += np.dot(f[i],X[i])
		dr3 +=f[i]
	r = nr3/dr3
	mu3 = r
	


	p1 = dr1/len(c1)
	p2 = dr2/len(c2)
	p3 = dr3/len(c3)


	sig1 = [0, 0]	
	for i in c1:
		sig1 += d[i]*(X[i] - mu1)*(np.transpose(X[i]-mu1))
	sig1 /=dr1
	cov1 = [[sig1[0],0],[0,sig1[1]]]

	sig2 = [0, 0]	
	for i in c2:
		sig2 += e[i]*(X[i] - mu2)*(np.transpose(X[i]-mu2))
	sig2 /=dr2
	cov2 = [[sig2[0],0],[0,sig2[1]]]

	sig3 = [0, 0]	
	for i in c3:
		sig3 += f[i]*(X[i] - mu3)*(np.transpose(X[i]-mu3))
	
	sig3 /=dr3

	cov3 = [[sig3[0],0],[0,sig3[1]]]


	def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
		def eigsorted(cov):
			vals, vecs = np.linalg.eigh(cov)
			order = vals.argsort()[::-1]
			return vals[order], vecs[:,order]
		
		if ax is None:
			ax = plt.gca()

		vals, vecs = eigsorted(cov)
		theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    	# Width and height are "full" widths, not radius
		width, height = 2 * nstd * np.sqrt(vals)
		ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
		ax.add_artist(ellip)
		return ellip


	if __name__ == '__main__':
		x, y = X.T
		plt.plot(x, y, 'y')
		plot_cov_ellipse(cov1, mu1, nstd=2, ax=None, color='red', alpha=0.5)
		plot_cov_ellipse(cov2, mu2, nstd=2, ax=None, color='lime',alpha=0.5)
		plot_cov_ellipse(cov3, mu3, nstd=2, ax=None, color='blue',alpha=0.5)
		imgname= "task3_gmm_iter"+str(z+1)+".jpg"
		plt.savefig(imgname)
		print(imgname+"  created...")























		