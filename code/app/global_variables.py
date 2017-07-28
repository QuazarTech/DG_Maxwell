import numpy as np
import arrayfire as af
from scipy import special as sp
from app import lagrange
from app import wave_equation
from utils import utils

gaussianNodesList = [ 
[ 0.57735027, -0.57735027],\
[-0.77459667, 0., 0.77459667],\
[-0.86113631, -0.33998104, 0.33998104, 0.86113631],\
[-0.90617985, -0.53846931, 0., 0.53846931, 0.90617985],\
[-0.93246951,-0.66120939,-0.23861919, 0.23861919, 0.66120939, 0.93246951],\
[-0.94910791, -0.74153119, -0.40584515, 0, 0.40584515, 0.74153119, 0.94910791],\
[-0.96028986, -0.79666648, -0.52553241, -0.18343464, 0.18343464, 0.52553241,\
  0.79666648, 0.96028986],\
[-0.96816024,-0.83603111,-0.61337143,-0.32425342 ,0.        , 0.32425342,
  0.61337143, 0.83603111, 0.96816024],\
[-0.97390653,-0.86506337,-0.67940957,-0.43339539,-0.14887434, 0.14887434,
  0.43339539, 0.67940957, 0.86506337, 0.97390653],\
[-0.97822866,-0.8870626 ,-0.73015201,-0.51909613, -0.26954316, 0.,
  0.26954316, 0.51909613, 0.73015201, 0.8870626 , 0.97822866],\
[-0.98156063,-0.90411726,-0.76990267,-0.58731795,-0.3678315, -0.12523341,
  0.12523341, 0.3678315,  0.58731795, 0.76990267, 0.90411726, 0.98156063],\
[-0.98418305,-0.9175984, -0.80157809,-0.64234934,-0.44849275,-0.23045832,
  0.,         0.23045832, 0.44849275, 0.64234934, 0.80157809, 0.9175984,
  0.98418305],\
[-0.98628381,-0.92843488,-0.82720132,-0.6872929 ,-0.51524864,-0.31911237,
 -0.10805495, 0.10805495, 0.31911237, 0.51524864, 0.6872929 , 0.82720132,
  0.92843488, 0.98628381],\
[-0.98799252,-0.93727339,-0.84820658,-0.72441773,-0.57097217,-0.39415135,
 -0.20119409, 0.        , 0.20119409, 0.39415135, 0.57097217, 0.72441773,
  0.84820658, 0.93727339, 0.98799252],\
[-0.98940093,-0.94457502,-0.8656312 ,-0.75540441,-0.61787624,-0.45801678,
 -0.28160355,-0.09501251, 0.09501251, 0.28160355, 0.45801678, 0.61787624,
  0.75540441, 0.8656312 , 0.94457502, 0.98940093],\
[-0.99057548,-0.95067552,-0.88023915,-0.781514  ,-0.65767116,-0.51269054,
 -0.35123176,-0.17848418, 0.        , 0.17848418, 0.35123176, 0.51269054,
  0.65767116, 0.781514,   0.88023915, 0.95067552, 0.99057548],\
[-0.99156517,-0.95582395,-0.89260247,-0.80370496,-0.69168704,-0.55977083,
 -0.41175116,-0.25188623,-0.08477501, 0.08477501, 0.25188623, 0.41175116,
  0.55977083, 0.69168704, 0.80370496, 0.89260247, 0.95582395, 0.99156517],\
[-0.99240684,-0.96020815,-0.9031559 ,-0.82271466,-0.72096618,-0.6005453,
 -0.46457074,-0.3165641 ,-0.16035865, 0.        , 0.16035865, 0.3165641,
  0.46457074, 0.6005453 , 0.72096618, 0.82271466, 0.9031559 , 0.96020815,
  0.99240684],\
[-0.9931286 ,-0.96397193,-0.91223443,-0.83911697,-0.74633191,-0.63605368,
 -0.510867  ,-0.37370609,-0.22778585,-0.07652652, 0.07652652, 0.22778585,
  0.37370609, 0.510867 ,  0.63605368, 0.74633191, 0.83911697, 0.91223443,
  0.96397193, 0.9931286 ],\
[-0.99375217,-0.96722684,-0.92009933,-0.85336337,-0.76843996,-0.6671388,
 -0.55161884,-0.42434212,-0.28802132,-0.14556185, 0.        , 0.14556185,
  0.28802132, 0.42434212, 0.55161884, 0.6671388 , 0.76843996, 0.85336336,
  0.92009933, 0.96722684, 0.99375217],\
[-0.99429458,-0.97006051,-0.92695676,-0.86581258,-0.7878168 ,-0.69448726,
 -0.5876404 ,-0.46935584,-0.34193582,-0.20786043,-0.06973927, 0.06973927,
  0.20786043, 0.34193582, 0.46935584, 0.5876404 , 0.69448726, 0.78781681,
  0.86581258, 0.92695677, 0.9700605 , 0.99429459],\
[-0.99476934,-0.97254246,-0.9329711 ,-0.87675235,-0.8048884 ,-0.71866136,
 -0.61960988,-0.50950148,-0.39030104,-0.26413568,-0.13325682, 0.,
  0.13325682, 0.26413568, 0.39030104, 0.50950148, 0.61960988, 0.71866136,
  0.8048884 , 0.87675236, 0.93297109, 0.97254247, 0.99476934],\
[-0.99518723,-0.97472853,-0.93827459,-0.88641549,-0.82000201,-0.74012418,
 -0.64809365,-0.54542147,-0.43379351,-0.31504268,-0.19111887,-0.06405689,
  0.06405689, 0.19111887, 0.31504268, 0.43379351, 0.54542147, 0.64809365,
  0.74012419, 0.82000199, 0.88641553, 0.93827455, 0.97472856, 0.99518722],\
[-0.99555687,-0.97666418,-0.94297429,-0.89499218,-0.83344255,-0.75925929,
 -0.67356636,-0.57766293,-0.47300273,-0.36117231,-0.24386688,-0.12286469,
  0.        , 0.12286469, 0.24386688, 0.36117231, 0.47300273, 0.57766293,
  0.67356637, 0.75925926, 0.83344263, 0.894992  , 0.94297457, 0.97666392,
  0.99555697],\
[-0.9958858 ,-0.97838518,-0.94715938,-0.90263762,-0.84544607,-0.7763859,
 -0.69642727,-0.60669229,-0.50844072,-0.40305176,-0.29200484,-0.17685882,
 -0.05923009, 0.05923009, 0.17685882, 0.29200484, 0.40305176, 0.50844071,
  0.60669229, 0.69642726, 0.77638595, 0.84544594, 0.90263786, 0.94715907,
  0.97838545, 0.9958857 ],\
[-0.99617909,-0.97992388,-0.9509002 ,-0.90948246,-0.85620791,-0.79177161,
 -0.71701349,-0.63290797,-0.54055157,-0.44114825,-0.3359939 ,-0.22645937,
 -0.11397259, 0.        , 0.11397259, 0.22645937, 0.3359939 , 0.44114825,
  0.54055156, 0.63290797, 0.71701347, 0.79177164, 0.85620791, 0.90948232,
  0.95090055, 0.97992348, 0.99617926],\
[-0.9964411 ,-0.98130662,-0.95425567,-0.91563526,-0.86589166,-0.80564157,
 -0.73561086,-0.65665109,-0.56972047,-0.47587423,-0.37625152,-0.27206163,
 -0.16456928,-0.05507929, 0.05507929, 0.16456928, 0.27206163, 0.37625152,
  0.47587423, 0.56972047, 0.65665109, 0.73561088, 0.80564137, 0.86589252,
  0.91563302, 0.95425928, 0.98130316, 0.9964425 ],\
[-0.99666983,-0.98257142,-0.95725386,-0.92120547,-0.8746236 ,-0.8181913,
 -0.75246112,-0.67821491,-0.59628174,-0.50759296,-0.41315289,-0.31403164,
 -0.21135229,-0.10627823, 0.        , 0.10627823, 0.21135229, 0.31403164,
  0.41315289, 0.50759295, 0.5962818 , 0.67821454, 0.75246285, 0.81818548,
  0.87463781, 0.92118023, 0.9572856 , 0.9825455 , 0.99667944],\
[-0.99687009,-0.98373011,-0.95994813,-0.92625607,-0.88253118,-0.82957654,
 -0.76777475,-0.69785091,-0.62052616,-0.53662415,-0.44703377,-0.35270473,
 -0.25463693,-0.15386991,-0.05147184, 0.05147184, 0.15386991, 0.25463693,
  0.35270473, 0.44703377, 0.53662415, 0.62052618, 0.6978505 , 0.76777742,
  0.82956577, 0.88256052, 0.92620007, 0.96002184, 0.98366815, 0.99689347]

]

LGL_list = [ \
[-1.0,1.0],                                                               \
[-1.0,0.0,1.0],                                                           \
[-1.0,-0.4472135955,0.4472135955,1.0],                                    \
[-1.0,-0.654653670708,0.0,0.654653670708,1.0],                            \
[-1.0,-0.765055323929,-0.285231516481,0.285231516481,0.765055323929,1.0], \
[-1.0,-0.830223896279,-0.468848793471,0.0,0.468848793471,0.830223896279,  \
1.0],                                                                     \
[-1.0,-0.87174014851,-0.591700181433,-0.209299217902,0.209299217902,      \
0.591700181433,0.87174014851,1.0],                                        \
[-1.0,-0.899757995411,-0.677186279511,-0.363117463826,0.0,0.363117463826, \
0.677186279511,0.899757995411,1.0],                                       \
[-1.0,-0.919533908167,-0.738773865105,-0.47792494981,-0.165278957666,     \
0.165278957666,0.47792494981,0.738773865106,0.919533908166,1.0],          \
[-1.0,-0.934001430408,-0.784483473663,-0.565235326996,-0.295758135587,    \
0.0,0.295758135587,0.565235326996,0.784483473663,0.934001430408,1.0],     \
[-1.0,-0.944899272223,-0.819279321644,-0.632876153032,-0.399530940965,    \
-0.136552932855,0.136552932855,0.399530940965,0.632876153032,             \
0.819279321644,0.944899272223,1.0],                                       \
[-1.0,-0.953309846642,-0.846347564652,-0.686188469082,-0.482909821091,    \
-0.249286930106,0.0,0.249286930106,0.482909821091,0.686188469082,         \
0.846347564652,0.953309846642,1.0],                                       \
[-0.999999999996,-0.959935045274,-0.867801053826,-0.728868599093,         \
-0.550639402928,-0.342724013343,-0.116331868884,0.116331868884,           \
0.342724013343,0.550639402929,0.728868599091,0.86780105383,               \
0.959935045267,1.0],                                                      \
[-0.999999999996,-0.965245926511,-0.885082044219,-0.763519689953,         \
-0.60625320547,-0.420638054714,-0.215353955364,0.0,0.215353955364,        \
0.420638054714,0.60625320547,0.763519689952,0.885082044223,               \
0.965245926503,1.0],                                                      \
[-0.999999999984,-0.9695680463,-0.899200533072,-0.792008291871,           \
-0.65238870288,-0.486059421887,-0.299830468901,-0.101326273522,           \
0.101326273522,0.299830468901,0.486059421887,0.652388702882,              \
0.792008291863,0.899200533092,0.969568046272,0.999999999999]]


for idx in np.arange(len(LGL_list)):
	LGL_list[idx] = np.array(LGL_list[idx], dtype = np.float64)
	LGL_list[idx] = af.interop.np_to_af_array(LGL_list[idx])

for idx in np.arange(len(gaussianNodesList)):
	gaussianNodesList[idx] = np.array(gaussianNodesList[idx], \
										dtype = np.float64)
	gaussianNodesList[idx] = af.interop.np_to_af_array(gaussianNodesList[idx])

x_nodes         = af.interop.np_to_af_array(np.array([[-1., 1.]]))
N_LGL           = 16
xi_LGL          = None
lBasisArray     = None
lobatto_weights = None
N_Elements      = None
element_nodes   = None


def populateGlobalVariables(Number_of_LGL_pts = 8, Number_of_Gauss_nodes = 9,
							Number_of_elements = 10):
	'''
	Initialize the global variables.
	Parameters
	----------
	N_LGL : int
			Number of LGL points.
			Declares the number and the value of
	
	N_gauss : int
			  Number of gaussian nodes required.
	
	'''
	
	global N_LGL
	global xi_LGL
	global lBasisArray
	global lobatto_weights
	N_LGL       = Number_of_LGL_pts
	xi_LGL      = lagrange.LGL_points(N_LGL)
	lBasisArray = af.interop.np_to_af_array( \
		lagrange.lagrange_basis_coeffs(xi_LGL))
	
	lobatto_weights = af.interop.np_to_af_array(\
		lobatto_weight_function(N_LGL, xi_LGL)) 
	
	global N_Elements
	global element_nodes
	N_Elements       = Number_of_elements
	element_size     = af.sum((x_nodes[0, 1] - x_nodes[0, 0]) / N_Elements)
	elements_xi_LGL  = af.constant(0, N_Elements, N_LGL)
	elements         = utils.linspace(af.sum(x_nodes[0, 0]), \
		af.sum(x_nodes[0, 1] - element_size), N_Elements)
	
	np_element_array = np.concatenate((af.transpose(elements), 
						   af.transpose(elements + element_size)))
	
	element_array = (af.transpose(af.interop.np_to_af_array(np_element_array)))
	element_nodes = af.transpose(wave_equation.mappingXiToX(\
										af.transpose(element_array), xi_LGL))
	
	
	global u
	global time
	u_init     = np.e ** (-(element_nodes) ** 2 / 0.4 ** 2)
	time       = utils.linspace(0, 2, 20)
	u          = af.constant(0, (N_Elements), N_LGL, time.shape[0])
	u[:, :, 0] = u_init
	
	global c
	c = 1.0
	
	return


def gaussian_weights(N, i):
	'''
	Returns the gaussian weights for :math: `N` Gaussian Nodes at index
	 :math: `i`.
	
	Parameters
	----------
	N     : int
			Number of Gaussian nodes for which the weight is t be calculated.
			
	i     : int
			Index for which the Gaussian weight is required.
	
	Returns
	-------
	gaussian_weight : double 
					  The gaussian weight.
	
	'''
	
	gaussian_weight  = 2 / ((1 - ((af.sum(gaussianNodesList[N - 2][i]))) ** 2) \
	* (np.polyder(sp.legendre(N))(af.sum(gaussianNodesList[N - 2][i]))) ** 2)
	
	
	return gaussian_weight


def lobatto_weight_function(n, x):
	'''
	Calculates and returns the weight function for an index :math:`n`
	and points :math: `x`.
	
	:math::
		w_{n} = \\frac{2 P(x)^2}{n (n - 1)},
		Where P(x) is $ (n - 1)^th $ index.
	
	Parameters
	----------
	n : int
		Index for which lobatto weight function
	
	x : arrayfire.Array
		1D array of points where weight function is to be calculated.
	
	.. lobatto weight function -
	https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules
	
	Returns
	-------
	An array of lobatto weight functions for the given :math: `x` points
	and index.
	
	'''
	P = sp.legendre(n - 1)
	
	return (2 / (n * (n - 1)) / (P(x))**2)


def lagrange_basis_function():
	'''
	Funtion which calculates the value of lagrange basis functions over LGL
	nodes.
	
	Returns
	-------
	L_i    : arrayfire.Array [N 1 1 1]
			 The value of lagrange basis functions calculated over the LGL
			 nodes.
	'''
	xi_tile    = af.transpose(af.tile(xi_LGL, 1, N_LGL))
	power      = af.flip(af.range(N_LGL))
	power_tile = af.tile(power, 1, N_LGL)
	xi_pow     = af.arith.pow(xi_tile, power_tile)
	index      = af.range(N_LGL)
	L_i        = af.blas.matmul(lBasisArray[index], xi_pow)
	
	return L_i
