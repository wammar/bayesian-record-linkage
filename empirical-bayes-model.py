import re
import time
from datetime import datetime
import io
import sys
import argparse
from collections import defaultdict
#import Levenshtein
import scipy
import scipy.stats
import scipy.special
import random
import csv
import math
import numpy
import fileinput
import cProfile
import threading

# feature prefixes
STRING_SIMILARITY = 'similarity'
SIMILAR_MINUS_DIFFERENT_CHARS_COUNT = 'similar-diff'
DIFFERENT_CHARS = 'diffs'
SUBSTRING = 'sub'
RELATIVE_NUMERIC_DIFF = 'rel-diff'

# observed/input variables
a = 1     # first parameter of the beta prior
b = 1     # second parameter of the beta prior

c = defaultdict(float)    # feature weights
l2_strength = 0.1        # l2 regularization strength
learning_rate = 0.1       # learning rate for gradient ascent

x = []    # list of observed records, each of which is another list, all internal lists should be consistent with field_types

m = 450   # number of latent records
field_types = [str, str, int, int, int] # field types for each observed/latent record
max_iter_count = 10
threads_count = 1

# latent variables
y = []             # list of latent records, each of which is another list, 
                   # all internal lists should be consistent with field_types
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
total_time, avg_time, prev_timestamp = datetime.now()-datetime.now(), None, datetime.now()

# must be called after set_observables() has been called
def fit_feature_weights(args):
  global c

  # only fit feature weights when training pairs of similar observed records are provided
  if not args.training_pairs: return 

  # read similar pairs
  original, distorted = [], []
  for line in open(args.training_pairs):
    # indexes of pairs of similar records are written in each line, separated with a comma 
    index1, index2 = line.strip().split(',')
    # indexes in the file are one-based, but we use zero-based indexes internally, so:
    index1, index2 = int(index1)-1, int(index2)-1
    # first, assume index2 is a noisy copy of index1    
    original.append( list(x[index1]) )
    distorted.append( list(x[index2]) )
    # then, assume index1 is a noisy copy of index2
    original.append( list(x[index2]) )
    distorted.append( list(x[index1]) )
  assert len(original) == len(distorted)
    
  # use gradient descent to optimize feature weights
  gradient_ascent_converged = False
  iter_count = 0
  while not gradient_ascent_converged:
    log_likelihood = 0.0
    gradient = defaultdict(float)

    # compute likelihood and gradient
    for i in xrange(len(original)):
      for l in xrange(len(field_types)):
        # only use distorted values
        if distorted[i][l] == original[i][l]: continue

        # positively enforce the observed distortion
        x_value, y_value = distorted[i][l], original[i][l]
        log_likelihood += score(l, x_value, y_value)
        active_features = fire_features(l, x_value, y_value)
        for feature_id, feature_value in active_features.iteritems():
          gradient[feature_id] += feature_value

        # positively enforce no distortion
        log_likelihood += score(l, y_value, y_value)
        active_features = fire_features(l, y_value, y_value)
        for feature_id, feature_value in active_features.iteritems():
          gradient[feature_id] += feature_value

        # compute the partition function for the distortion model conditional on y_value
        log_partition = -300
        for other_x_value in x_domain[l]:
          log_partition = numpy.logaddexp(log_partition, score(l, other_x_value, y_value))
          
        # negative enforcement (for two values: the distorted and the original)
        log_likelihood -= 2 * log_partition
        for other_x_value in x_domain[l]:
          distortion_prob = math.e ** (score(l, other_x_value, y_value) - log_partition)
          other_active_features = fire_features(l, other_x_value, y_value)
          for feature_id, feature_value in other_active_features.iteritems():
            gradient[feature_id] -= 2 * distortion_prob * feature_value
          # hopefully, these probabilities will keep increasing as we fit c
          #if other_x_value == distorted[i][l]: print 'p(', distorted[i][l], '|', original[i][l], ')=', distortion_prob
    
    # l2 regularization
    for feature_id, feature_weight in c.iteritems():
      log_likelihood -= l2_strength * (feature_weight**2)
      gradient[feature_id] -= 2 * l2_strength * feature_weight

    # done computing likelihood and gradient
    for feature_id in c.keys():
      c[feature_id] += gradient[feature_id] * learning_rate # check the sign

    print
    print 'completed gradient ascent iteration #', iter_count 
    #print 'gradient = ', gradient
    print '|gradient|_2 = ', sum([g*g for g in gradient.values()])
    print 'regularized log-likelihood = ', log_likelihood
    #print 'c = ', c
    fileinput.input()

    # check convergence
    iter_count += 1
    if iter_count == 10:
      gradient_ascent_converged = True

  # congrat! gradient ascent has converged
  print 'done with gradient ascent'

def set_observables(args):
  global x_domain, x

  # read the observed records to set x
  if args.observed_records:
    observed_records_file = io.open(args.observed_records, encoding='utf8', mode='r')
    observed_records_reader = csv.reader(observed_records_file, delimiter=',')
    headers = observed_records_reader.next()
    assert len(headers) == len(field_types)+1

    for observed_record in observed_records_reader:
      for i in range(1, len(observed_record)):
        if field_types[i-1] == int:
          observed_record[i] = int(observed_record[i])
        elif field_types[i-1] == float:
          observed_records[i] = float(observed_record[i])
        elif field_types[i-1] == str:
          pass
        else:
          assert False
      x.append(observed_record[1:])
      #print observed_record[1:]
  else:
    x.append( ['kartik', 10, 5.0] )
    x.append( ['waleed', 20, 10.5] )
    x.append( ['kartek', 11, 5.1] )

  print '|observed records| = ', len(x)
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

def compute_Lambda_posteriors(posteriors, flags, i, thread_id):
  for j in xrange(m):
    # compute the posterior of p(lambda_i = j | everything else)
    posterior = 1.0
    #print 'j=', j
    if j % threads_count != thread_id:
      continue
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
    posteriors[j] = posterior
    flags[j] = True
    #print 'thread_id=', thread_id, ', flags=', flags[:threads_count]

def resample_Lambda(i):
  global Lambda, Lambda_inverse

  #print 'inside resample_Lambda(', i, ')'
  # first, compute the posterior distribution of lambda_i
  posteriors = [0.0 for index in xrange(m)]
  flags = [False for index in xrange(m)]
  if threads_count != 1:
    for thread_id in xrange(threads_count):
      workload = (posteriors, flags, i, thread_id)
      t = threading.Thread(target=compute_Lambda_posteriors, args=workload)
      t.start()
    # sync
    while flags[0] == False or len(set(flags)) > 1: 
      pass
  else:
    thread_id = 0
    compute_Lambda_posteriors(posteriors, flags, i, thread_id)

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
  #print 'leaving resample_Lambda(', i, ')'
  
def resample_z(i,l):
  global z

  #print 'inside resample_z(', i, ', ', l, ')'
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
  #print 'leaving resample_z(', i, ', ', l, ')'
  
def resample_y(i,l):
  global y

  #print 'inside resample_y(', i, ',', l, ')'

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
  #print 'inside resample_y(', i, ',', l, ')'

def check_convergence():
  global iter_count, prev_timestamp, total_time, avg_time
  if iter_count %100 == 0:
    print 'iter_count = ', iter_count
    print_linkage_structure()
  print '=================== END OF ITERATION', iter_count, ' ===================='
  if iter_count:
    total_time += datetime.now() - prev_timestamp
    avg_time = total_time / iter_count
    print 'avg iteration time = ', avg_time.total_seconds()
  prev_timestamp = datetime.now()
  iter_count += 1
  return iter_count > max_iter_count
    
def print_linkage_structure():
  print 'CURRENT LINKAGE STRUCTURE:'
  print
#  print 'x=', x
#  print 'y=', y
#  print 'z=', z
#  print 'Lambda=', Lambda
#  print
  for j in xrange(len(Lambda_inverse)):
    if len(Lambda_inverse[j]) > 0:
      print 'latent_index={}\nobserved_indexes={}\nlatent_record={}\nobserved_records={}\n\n'.format(j,
                                                                                                 Lambda_inverse[j],
                                                                                                 y[j],
                                                                                                 '\n'.join([str(x[i]) for i in Lambda_inverse[j]])) 
  print 'END OF CURRENT LINKAGE STRUCTURE ====================='

def levenshtein_ratio(s1, s2):
  return 1.0 - 1.0 * levenshtein(s1, s2) / max(len(s1), len(s2))

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
 
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
 
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
 
    return previous_row[-1]

def fire_features(l, x_value, y_value):
  active_features = {}

  # fire string similarity feature, for strings of lengths > 1
  if field_types[l] == int or field_types[l] == float:
    active_features['{}={}:{}'.format(RELATIVE_NUMERIC_DIFF, round(1.0*abs(x_value - y_value)/max(x_domain[l]), 1), field_types[l])] = 1.0
    x_value, y_value = str(x_value), str(y_value)
  else:
    x_value, y_value = x_value.lower(), y_value.lower()
  if len(x_value) > 1 and len(y_value) > 1:
    active_features['{}={}:{}'.format(STRING_SIMILARITY, 3 * round(levenshtein_ratio(x_value, y_value)/3, 1), field_types[l])] = 1.0
  
  # find different and similar characters
  x_chars, y_chars = set(x_value), set(y_value)
  diff_chars = (x_chars-y_chars)|(y_chars-x_chars)
  similar_chars = x_chars & y_chars
  active_features['{}={}:{}'.format(SIMILAR_MINUS_DIFFERENT_CHARS_COUNT, len(similar_chars)-len(diff_chars), field_types[l])] = 1.0
  if len(diff_chars) >= 1 and len(diff_chars) <= 2 and len(diff_chars) < len(similar_chars):
    active_features['{}={}:{}'.format(DIFFERENT_CHARS, ''.join(diff_chars), field_types[l])] = 1.0
    pass

  # substrings
  if x_value.find(y_value) >= 0 or y_value.find(x_value) >= 0:
    active_features['{}=1:{}'.format(SUBSTRING, field_types[l])] = 1.0
    pass

  return active_features

def score(l, x_value, y_value):
  total_score = 0.0
  for feature_id, feature_value in fire_features(l, x_value, y_value).iteritems():
    total_score += feature_value * c[feature_id]
  return total_score

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
  argparser.add_argument("-tp", "--training-pairs", required=False, help="optimize feature weights to maximize the likelihood of the distortion model based on these pairs of similar records")
  args = argparser.parse_args()
  return args

def main():
  args = parse_arguments()
  set_observables(args)
  fit_feature_weights(args)
  precompute_distortions()
  precompute_empiricals()
  init_latents()
  while True:

    for i in xrange(len(x)):
      resample_Lambda(i)
      #print 'Lambda[', i, ']=', Lambda[i]
      for l in xrange(len(field_types)):
        resample_z(i,l)
        #print 'z[', i, '][', l, ']=', z[i][l]
        
    for j in xrange(m):
      for l in xrange(len(field_types)):
        resample_y(j,l)
        #print 'y[', j, '][', l, ']=', y[j][l]
        #print_linkage_structure()

    if check_convergence(): break

  print 'CONVERGED'
  print_linkage_structure()

if __name__ == "__main__":
  #main()
  cProfile.run('main()')

