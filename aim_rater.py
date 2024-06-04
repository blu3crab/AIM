###############################################################################
# aim_rater - compare 2 lists of predictions
#
# return->
#   matches,
#   partial_matches,
#   exact_match_percentage,
#   partial_match_percentage

#!pip install fuzzywuzzy

# TODO!
# /usr/local/lib/python3.10/dist-packages/fuzzywuzzy/fuzz.py:11: UserWarning: Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning
#   warnings.warn('Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning')
###############################################################################
# fixes multiple groundtruth word matching e.g. said, said, said, said
import fuzzywuzzy.fuzz

def rate_match(groundtruth_list, prediction_list, print_list=True):
  print(f"groundtruth_list => {groundtruth_list}")
  print(f"prediction list => {prediction_list}")

  # Partial match threshold (experiment with this value)
  PARTIAL_MATCH_THRESHOLD = 80

  # Find matches, partial matches, and mismatches
  match_list = []
  partial_match_list = []
  mismatch_list = []

  # Initialize counts and tracking dictionaries
  matches = 0
  partial_matches = 0
  groundtruth_counts = {word: groundtruth_list.count(word) for word in groundtruth_list}
  predicted_counts = {word: prediction_list.count(word) for word in prediction_list}

  for word in prediction_list:
      best_match_found = False
      partial_match_found = False

      for groundtruth_word in groundtruth_list:
          ratio = fuzzywuzzy.fuzz.ratio(word, groundtruth_word)

          if ratio == 100 and groundtruth_counts[groundtruth_word] > 0:
              matches += 1
              match_list.append(word)
              groundtruth_counts[groundtruth_word] -= 1  # Decrement count for matched word
              best_match_found = True
              break

      if not best_match_found:
          for groundtruth_word in groundtruth_list:
              ratio = fuzzywuzzy.fuzz.ratio(word, groundtruth_word)

              if ratio >= PARTIAL_MATCH_THRESHOLD and groundtruth_counts[groundtruth_word] > 0:
                  partial_matches += 1
                  partial_match_list.append(word)
                  groundtruth_counts[groundtruth_word] -= 1
                  partial_match_found = True
                  break
      if not best_match_found and not partial_match_found:
          mismatch_list.append(word)

  # Calculate percentages more accurately
  exact_match_percentage = (matches / len(groundtruth_list)) * 100 if groundtruth_list else 0
  partial_match_percentage = (partial_matches / len(groundtruth_list)) * 100 if groundtruth_list else 0

  # Print the results
  print("Groundtruth list length:", len(groundtruth_list))
  print("Prediction list length:", len(prediction_list))
  print(f"Exact match percentage: {exact_match_percentage}% with exact match count {matches}")
  print(f"Partial match percentage: {partial_match_percentage}% with exact match count {partial_matches}")
  #print("Partial match percentage:", partial_match_percentage, "%")
  if print_list:
    print("\nMatching words:")
    print(match_list)
    print("\nPartially Matching words:")
    print(partial_match_list)
    print("\nMismatched words:")
    print(mismatch_list)

  return matches, partial_matches, exact_match_percentage, partial_match_percentage