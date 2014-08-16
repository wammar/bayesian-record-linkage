import re
import time
import io
import sys
import argparse
from collections import defaultdict
import Levenshtein
import scipy
import scipy.stats
import scipy.special
import random

# observed/input variables
a = 1     # first parameter of the beta prior
b = 1     # second parameter of the beta prior
c = [-1.0, -1.0] # feature weights -- TODO: learn this from data
x = []    # list of observed records, each of which is another list, all internal lists should be consistent with field_types
m = 0     # number of latent records
field_types = [str, int, float] # field types for each observed/latent record
max_iter_count = 100

# latent variables
y = []             # list of latent records, each of which is another list, all internal lists should be consistent with field_types
Lambda = []        # list of "pointers" to the latent record that corresponds to an observed record
                   # Lambda[i] = j <=> observed record x[i] was generated from latent record y[j]
z = []             # list of lists of booleans. the size of internal lists should match field_types
                   # z[j][l] = true <=> x[j][l] has been noisily generated from y[Lambda[j]][l]
                   # note that z[j][l] = false implies that x[j][l] = y[Lambda[j]][l], but
                   #           z[j][l] = true doesn't imply that x[j][l] != y[Lambda[j]][l]

# deterministically-dependent variables
Lambda_inverse = []# list of sets. for each latent record y[i], Lambda_inverse[i] gives the indexes {j} of
                   # aligned observed records x[j]
                   # Lambda_inverse[j] = set([i1, i2, i3]) <=> Lambda[i1] = Lambda[i2] = Lambda[i3] = j, Lambda[else] != j 
empirical = []     # the empirical distribution for each field in the observed records
                   # empirical[l] is a dictionary that maps a field value to the percentage of records with that value
distortion = {}    # distortion[(l,y')] => the distortion distribution of x' | y' for field l
                   #                       which is a defaultdict
x_domain = []      # list of sets. x_domain[l] is the set of observed values at field l 
iter_count = 0 

def set_observables(args):
  global x_domain, x, m
  # set number of unique latent records
  m = 2

  # read the observed records to set x
  x.append( ['kartik', 10, 5.0] )
  x.append( ['waleed', 20, 10.5] )
  x.append( ['kartek', 11, 5.1] )
  
  # populate x_domain
  x_domain = [set([rec[l] for rec in x]) for l in xrange(len(field_types))] 
  
def init_latents():
  global y, Lambda, Lambda_inverse, z
  # copy the latent records from the observed records verbatim
  y = [list(x[i]) for i in xrange(m)]

  # each observed points at the corresponding latent; if the number of unique latents (m) < len(x), then use m-1
  Lambda = [min(i, m-1) for i in xrange(len(x))]
  Lambda_inverse = [set() for i in xrange(m)]
  for i in xrange(len(x)):
    Lambda_inverse[ Lambda[i] ].add(i)

    # initially, no distortion happens
    z.append([])
    for l in xrange(len(field_types)):
      if i < m:
        z[i].append(False)
      else:
        # except when Lambda[i] != i
        z[i].append(True)

# assumption: alist contains nonnegative elements only
def list_to_unit_vector_insitu(alist):
  summation = sum(alist)
  if summation == 0:
    print_linkage_structure()
  assert summation != 0
  for i in xrange(len(alist)):
    alist[i] /= summation

# assumption: adict contains nonnegative values only
def dict_to_unit_vector_insitu(adict):
  summation = sum(adict.values())
  for k in adict.keys():
    adict[k] /= summation
  
# assumption: multinomial is a list that sums to one and has non-negative elements
def sample_index_from_multinomial_list(multinomial):
  mark = random.random()
  commulative = 0.0
  for i in xrange(len(multinomial)):
    commulative += multinomial[i]
    if commulative > mark:
      return i
  assert(False)

# assumption: multinomial is a dict with values that sums to one and has non-negative elements
def sample_key_from_multinomial_dict(multinomial):
  mark = random.random()
  commulative = 0.0
  for k in multinomial.keys():
    commulative += multinomial[k]
    if commulative > mark:
      return k

def resample_Lambda(i):
  global Lambda, Lambda_inverse

  print 'inside resample_Lambda(', i, ')'
  # first, compute the posterior distribution of lambda_i
  posteriors = []
  for j in xrange(m):
    # compute the posterior of p(lambda_i = j | everything else)
    posterior = 1.0
    #print 'j=', j
    for l in xrange(len(field_types)):
      #print 'l=', l
      if z[i][l] == False:        
        #print 'z[i][l]=', z[i][l]
        #print 'x[i][l]=', x[i][l], ', y[j][l]=', y[j][l]
        delta = 1.0 if x[i][l] == y[j][l] else 0.0
        #print 'delta=', delta
        posterior *= scipy.special.beta(a, b+1) * delta
        #print 'posterior *=', scipy.special.beta(a, b+1) * delta, ' = ', posterior
      else:
        #print 'z[i][l]=', z[i][l]
        posterior *= scipy.special.beta(a+1, b) * distortion[(l, y[j][l])][x[i][l]]
    posteriors.append(posterior)

  if sum(posteriors) == 0: exit(1)
  # normalize the posteriors
  list_to_unit_vector_insitu(posteriors)
  
  # now that we computed the posteriors, sample a value for lambda_i 
  old_lambda_i = Lambda[i]
  
  lambda_i = sample_index_from_multinomial_list(posteriors)
  Lambda[i] = lambda_i
  
  # update inverse lambda
  Lambda_inverse[old_lambda_i] -= set([i])
  Lambda_inverse[Lambda[i]] |= set([i])
  print 'leaving resample_Lambda(', i, ')'
  
def resample_z(i,l):
  global z

  print 'inside resample_z(', i, ', ', l, ')'
  # first, compute the posterior distribution of z_{i,l}
  delta = 1.0 if x[i][l] == y[Lambda[i]][l] else 0.0
  posteriors = []
  posterior_prob_of_z_eq_zero = scipy.special.beta(a, b+1) * delta
  posteriors.append(posterior_prob_of_z_eq_zero)
  posterior_prob_of_z_eq_one  = scipy.special.beta(a+1, b) * distortion[(l, y[Lambda[i]][l])][x[i][l]]
  posteriors.append(posterior_prob_of_z_eq_one)
  
  # normalize the posteriors
  list_to_unit_vector_insitu(posteriors)

  # now sample a value for z_{i,l}
  if sample_index_from_multinomial_list(posteriors) == 0:
    z[i][l] = False
  else:
    z[i][l] = True
  print 'leaving resample_z(', i, ', ', l, ')'
  
def resample_y(i,l):
  global y

  print 'inside resample_y(', i, ',', l, ')'

  # find the indexes of observed records which are currently aligned to this latent record
  aligned_observed_record_indexes = Lambda_inverse[i]

  # first, compute posteriors
  posteriors = defaultdict(float)
  for v in x_domain[l]:
    posteriors[v] = 1.0
    for j in aligned_observed_record_indexes:
      if z[j][l] == False:
        delta = 1.0 if x[j][l] == v else 0.0
        posteriors[v] *= float(scipy.special.beta(a, b+1)) * delta
      else:
        posteriors[v] *= float(scipy.special.beta(a+1, b)) * distortion[(l, v)][x[j][l]]

  # normalize posteriors
  dict_to_unit_vector_insitu(posteriors)
  
  # now, sample a value from the posterior
  y[i][l] = sample_key_from_multinomial_dict(posteriors)
  print 'inside resample_y(', i, ',', l, ')'

def check_convergence():
  global iter_count
  print '=================== END OF ITERATION', iter_count, ' ===================='
  iter_count += 1
  return iter_count > max_iter_count
    
def print_linkage_structure():
#  print 'CURRENT LINKAGE STRUCTURE:'
  print
  print 'x=', x
  print 'y=', y
  print 'z=', z
  print 'Lambda=', Lambda
  print
#  for j in xrange(len(Lambda_inverse)):
#    if len(Lambda_inverse[j]) > 0:
#      print 'latent_index={}\nobserved_indexes={}\nlatent_record={}\nobserved_records={}\n\n'.format(j,
#                                                                                                 Lambda_inverse[j],
#                                                                                                 y[j],
#                                                                                                 '\n'.join([str(x[i]) for i in Lambda_inverse[j]])) 
#  print 'END OF CURRENT LINKAGE STRUCTURE ====================='

def score(l, x_value, y_value):
  if field_types[l] == str:
    return c[0] * Levenshtein.ratio(x_value, y_value)
  else:
    return c[1] * (x_value - y_value) * (x_value - y_value)

def precompute_distortions():
  global distortions
  for l in xrange(len(field_types)):
    for y_value in x_domain[l]:
      distortion[(l, y_value)] = defaultdict(float)
      for x_value in x_domain[l]:
        distortion[(l, y_value)][x_value] = score(l, x_value, y_value)
      sum_ly = sum(distortion[(l, y_value)].values())
      for x_value in x_domain[l]:
        distortion[(l, y_value)][x_value] /= sum_ly

def precompute_empiricals():
  global empirical
  for l in xrange(len(field_types)):
    empirical.append(defaultdict(float))
  for i in xrange(len(x)):
    for l in xrange(len(field_types)):
       empirical[l][x[i][l]] += 1.0
  for l in xrange(len(field_types)):
    sum_l = sum(empirical[l].values())
    for value in empirical[l].keys():
      empirical[l][value] /= sum_l

def parse_arguments():
  # parse/validate arguments
  argparser = argparse.ArgumentParser()
  argparser.add_argument("-or", "--observed-records", required=True, help="observed records")
  args = argparser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_arguments()
  set_observables(args)
  precompute_distortions()
  precompute_empiricals()
  init_latents()
  while True:

    print_linkage_structure()

    for i in xrange(len(x)):
      resample_Lambda(i)
      print 'Lambda[', i, ']=', Lambda[i]
      for l in xrange(len(field_types)):
        resample_z(i,l)
        print 'z[', i, '][', l, ']=', z[i][l]
        
    for j in xrange(m):
      for l in xrange(len(field_types)):
        resample_y(j,l)
        print 'y[', j, '][', l, ']=', y[j][l]
        print_linkage_structure()

    if check_convergence(): break

  print 'CONVERGED'
  print_linkage_structure()

