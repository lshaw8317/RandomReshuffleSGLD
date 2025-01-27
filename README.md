Code for the paper ''Random Reshuffling for Stochastic Gradient Langevin Dynamics'', Luke Shaw and Peter A. Whalley.

``ModelProblemCalculation.py`` contains the symbolic manipulations necessary to derive the expressions in section 4 of the paper, where we show SGLD-RM has variance error of order $h+Rh$, while SGLD-RR has error of order $h+(Rh)^2$. ``Gaussian1DExperiment.py`` generates samples and plots the experimental confirmation of the analytical Gaussian calculations.

``LogisticRegressionExperiments.py`` contains the code for the experiments in section 5 of the paper. One may run an experiment using the command ``runLRExp(expName,R,n_paths)`` where ``expName`` is a string and corresponds 
to one of the datafiles ('SimData', 'StatLog', 'CTG', 'Chess') provided, ``R`` is an integer number of batchs, and ``n_paths`` is an integer number of simultaneous stochastic gradient realisations to carry out. We provide HMC estimates (using $10^7$ samples) of the true parameter mean for the datasets (with fixed Gaussian prior assumed by the experiment class ``LRExp``). One may then plot results using ``plotter(expName,R)`` which uses the file saved by ``runLRExp``.
In order to see oscillations, we recommend $10^4$ paths, but for convergence plots 20 is sufficient.
