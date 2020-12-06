		----- GPU Binary Lens Code - Guide -----

- all code run with the following command
"sage -python XXXXXXX.py"

To perform a grid search run:
"Full_Search.py"

To perform a mcmc rountine on the output from the grid search run:
"emcee_minimised.py"

To plot histograms of the mcmc output run:
"mcmc_interpret.py"


This order of commands will perform the 3 stages required to determine a
maximised likelihood function for a given data set (MOA and/or OGLE).



--- INPUT parameters

All raw data should be located in:
/events/MOA_YYYY_LOC_XXX/

where:
YYYY is the year of observation
LOC is where the event is (BLG or LMC)
XXX is the events ID from the MOA database (3 digit number)

This folder should only contain the data to be passed into the code,
and nothing else. Any associated documents should reside within:
/events/MOA_YYYY_LOC_XXX/associated/



--- OUTPUT

Outputs a set of images from which the model parameter values can be determined
"output/"

_____________________________________________________________________________
			\\\\GRID SEARCH SECTION////


--- Before running the grid search

- Open up the "Full_Search.py" code
- Edit line 86 to be the location of your microlensing event folder
	e.g. "events/MOA_YYYY_LOC_XXX/"
- Edit Lines 114 (est_t0) and 116 (est_tE) to be sensible initial guesses for
  the t0 and tE parameters respectivly.



--- Other parameters that can be modified

Within the Define Variables section there is a list of parameters that
can be modified if required.

num_points = number of pixels in a single dimension on the magnification map

threads_per_block = Keep this smaller than and a multiple of "num_points"
(used for setting up GPU grid/block dimensions)

map_limits = half the dimension of the magnification map in eistien radius.

parameter search space (lines 94-112) = upper and lower limits and the step
sizes for the grid search space.



--- Output

Will output a list of different files used throughout the code.

"/current_working_file.txt"
 - information for other routines to use which tell them where information is

"/output/MOA_YYYY_LOC_XXX_glocal_vars.txt"
 - List of parameter values to be used by other routines

"/output/raw_data.png"
 - Image of the raw data being modelled.

"/output/MOA_TTT_LOC_XXX_best_fit_parameters.txt"
 - List of the grid search current best found parameters

"/output/d_q_map.txt"
 - information for the d/q minimised chi^2 map

"/output/mag_map.png"
 - A magnification map formed by the best parameters foudn from the grid search

"/output/complete_d_q_chi^2_map.png"
 - Map of the minimised d/q chi^2.

"/output/best_parameters_LC.png"
 - Plot of the best fit parameters LC on the raw data, doesn't always work.

_____________________________________________________________________________
			\\\\MCMC SECTION////

Generally just run the file and it will perform an MCMC routine on the data,
if grid search found a sensible value then the MCMC should minimise this
parameter set.



--- Parameters that can be modified

lines 24-26, can be modified to adjust the length and number of chains run.

ndim - [FIXED] as the number of parameters in the binary model (DON'T MODIFY)

nwalkers = number of walkers(chains) going through the routine, must be >3*ndim

burn_in = number of steps all walkers perform on first loop through.

sample_steps = long run of the MCMC all walkers perform to determine output of
the most likely parameters



--- Output

A single file which contains all the chain information from the long sample run
of the mcmc.
"output/mcmc_parameter_chains_MOA_YYYY_LOC_XXX.txt"


_____________________________________________________________________________
			\\\\MCMC SECTION////

Just run once to produce the histograms of the MCMC output

--- Input

No input required



--- Output

Produces a series of histograms for each individual parameters showing the
number of times the steps in the nwalkers fell within the area.

- Working on producing more 2D scatter plots
- Need to determine a histograms gaussian to derive the parameter value

All graphs are stored in:
"/output/mcmc_hist_of_XXX.png"

where XXX is the parameter name [d, q, rho, u0, phi, t0, tE]
