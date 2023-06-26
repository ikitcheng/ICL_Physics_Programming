!!! Warning !!!
If using Python 2.7: 
	1) uncomment content in '__init__.py file'. 
	2) replace Nc.NC() and MC.MC objects in test.py -> with just NC() and MC()
If using Python 3.6:
	1) it should run as it is. 


'Integrator' package:'
	Newton-Coates and Monte Carlo integration routines are located inside the 'integrator' folder.


Running the code:

1) open test.py
2) press f5 to run.


To change inputs:
1) scroll down to line 70 (User Inputs) 
2) Can change the integrand by changing the expression for 'f' in line 72.
3) Can change the limits of integration by changing the 'lim' list in line 73.
4) Can adjust relative accuracy by changing the second argument in Int(),MCInt() and metropInt() methods.

Note: the commented code at the bottom of test.py are for the plots used in the reports. The code combines the individual plots obtained by .plot() method. 