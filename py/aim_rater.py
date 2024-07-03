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

###############################################################################
def rate_xform_set(transform, groundtruth_list, prediction_list, transform_names, exact_match_values, partial_match_values, print_list=True):
    print(f"=========={transform}===========")
    match_count, partial_match_count, exact_match_percentage, partial_match_percentage = rate_match(groundtruth_list, eval(f'pred_{transform}_list'))
    exact_match_values.append(exact_match_percentage)
    partial_match_values.append(partial_match_percentage)
    transform_names.append(transform)

    return transform_names, exact_match_values, partial_match_values
###############################################################################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_matchA(transform_names, exact_match_values, partial_match_values):
    plt.figure(figsize=(10, 6))
    plt.bar(transform_names, exact_match_values, label='Exact Match')
    plt.bar(transform_names, partial_match_values, bottom=exact_match_values, label='Partial Match', color='C1')

    ax = plt.gca()
    for tick, exact_val, partial_val in zip(ax.get_xticks(), exact_match_values, partial_match_values):
        print(f"tick->{tick}, exact_val->{exact_val}, partial_val->{partial_val}")
        total_val = exact_val + partial_val

        total_val_str = "{:.2f}".format(total_val)
        exact_val_str = "{:.2f}".format(exact_val)
        partial_val_str = "{:.2f}".format(partial_val)
        ax.text(tick, total_val + 1.5, f'{total_val_str}%', ha='center', va='bottom')

        # use partial_val as X coord for exact_val, use eaxct_val+ as X coord for partial_val
        # NOTE: if partial_val is zero, histogram will be somewaht misaligned
        ax.text(tick, partial_val + 0.5, f'{exact_val_str}%', ha='center', va='bottom', color='orange')
        ax.text(tick, exact_val + 3.5, f'{partial_val_str}%', ha='center', va='top', color='green')

    plt.ylabel('Match Percentage')
    plt.xlabel('Transformation')
    plt.legend()
    plt.show()
###############################################################################
def test_exact_match():
    groundtruth_list = ['high', 'rate', 'makes', 'you', 'stand', 'out', 'come', 'get']
    pred_list1 = ['high', 'rate', 'makes', 'you', 'stand', 'out', 'come', 'get']  # 100% exact match
    pred_list2 = ['high', 'rate', 'generic', 'you', 'sit', 'prime', 'come', 'July']  # 50% exact match
    pred_list3 = [' entirely', 'different', 'words', 'list', 'zero', 'match', 'with', 'ground']  # 0% exact match

    match_count, partial_match_count, exact_match_percentage, partial_match_percentage = rate_match(groundtruth_list, pred_list1)
    match_count, partial_match_count, exact_match_percentage, partial_match_percentage = rate_match(groundtruth_list, pred_list2)
    match_count, partial_match_count, exact_match_percentage, partial_match_percentage = rate_match(groundtruth_list, pred_list3)
###############################################################################
def test_partial_match():
    groundtruth_list = ["sunny", "day", "breeze", "tulip", "light", "parrot", "summer", "cloud"]

    pred_list1 = ["sunnyer", "days", "breez", "tulipu", "lighti", "parot", "summmer", "cloudy"]  # 100% partial match

    pred_list2 = ["sunr", " breeze", " tulips", "lighthousei", "hous", "fire", "arrot", "cloudy"]  # 50% partial match

    pred_list3 = ["sunrise", "daylight", "plumbreeze", "tall", "tree", "partiel", "summarization", "close"]  # 0% partial match

    match_count, partial_match_count, exact_match_percentage, partial_match_percentage = rate_match(groundtruth_list, pred_list1)
    match_count, partial_match_count, exact_match_percentage, partial_match_percentage = rate_match(groundtruth_list, pred_list2)
    match_count, partial_match_count, exact_match_percentage, partial_match_percentage = rate_match(groundtruth_list, pred_list3)
###############################################################################
# groundtruth_list = ["sunnyday", "day", "breeze", "tulip", "light", "parrot", "summertime", "cloud"]
# pred_list1 = ["sunny", "days", "moonbreeze", "atulip", "lighti", "parnot", "summmer", "cloudy"] # 100% partial match
# match_count, partial_match_count, exact_match_percentage, partial_match_percentage = rate_match(groundtruth_list, pred_list1)

# NO partial match:
#                                   pred        groundtruth
#   prefix match of groundtruth -> sunny        sunnyday
#   suffix match f groundtruth -> moonbreeze    breeze
#   prefix match of groundtruth -> summer       summertime
#
# partial MATCH
#
#   prefix on pred
#   suffix on pred
#   letter subtitution within pred

# groundtruth_list => ['sunnyday', 'day', 'breeze', 'tulip', 'light', 'parrot', 'summertime', 'cloud']
# prediction list => ['sunny', 'days', 'moonbreeze', 'atulip', 'lighti', 'parnot', 'summmer', 'cloudy']
# Groundtruth list length: 8
# Prediction list length: 8
# Exact match percentage: 0.0% with exact match count 0
# Partial match percentage: 62.5% with exact match count 5
#
# Matching words:
# []
#
# Partially Matching words:
# ['days', 'atulip', 'lighti', 'parnot', 'cloudy']
#
# Mismatched words:
# ['sunny', 'moonbreeze', 'summmer']
###############################################################################

