import csv
import math

def get_exp(fname):
    with open(fname) as csvfile:
        reader = csv.DictReader(csvfile)
        output = []
        for row in reader:
            output.append(row)
    return output

def check_equal(reference, output, keys, ):
    for i in range(len(reference)):
        reference_row = reference[i]
        output_row = output[i]
        for key in keys:
            assert key in reference_row, "line %d key %s not in reference" % (i, key)
            assert key in output_row, "line %d key %s not in output" % (i, key)
            r = float(reference_row[key])
            o = float(output_row[key])
            assert math.isclose(r, o, rel_tol=1e-5), "line %d key %s reference: %s, output: %s" % (i, key, r, o)

def check_exactly_equal(reference, output, ):
    for i in range(len(reference)):
        reference_row = reference[i]
        output_row = output[i]
        for key in reference_row:
            assert key in output_row, key
            assert reference_row[key] == output_row[key], "%s reference: %s, output: %s" % (key, reference_row[key], output_row[key])
