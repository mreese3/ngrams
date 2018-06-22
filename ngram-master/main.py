#!/usr/bin/env python3
from NGram import NGram
import sys
import os

def run_model(trainingdata, N, strings, smoothing=False):
    '''
    Simple helper function to run each model configuration
    '''
    trainingdata.seek(0)
    model = NGram(N, trainingdata, smoothing)
    print(model.get_model_formatted())
    sys.stdout.write(os.linesep)
    print("Sentence Probability Evaluation")
    for s in strings:
        print("Log(p) = %08.4f".ljust(20) % model.sentence_probability(s)
                + "Sentence: <s> %s </s>" % s)
    sys.stdout.write(os.linesep)
    if smoothing:
        print("Generated Sentences")
        for s in model.generate_sentence(10):
            print("Log(p) = %08.4f".ljust(20) % model.sentence_probability(s)
                    + "Sentence: %s" % s)
        sys.stdout.write(os.linesep)


def main():
    strings = ["take the block on the green circle",
        "put the block on the circle on the red circle"]
    trainingfile = "training_data.txt"

    # Enable to use Lab 3 data
    #strings = ["I am eggs"]
    #trainingfile = "Lab3_example_data.txt"

    trainingdata = None

    try:
        trainingdata = open(trainingfile)
    except IOError as ioe:
        print("Could Not Open File: " + ioe.strerror)
        sys.exit(1)

    run_model(trainingdata, 1, strings)
    run_model(trainingdata, 2, strings)
    run_model(trainingdata, 1, strings, smoothing=True)
    run_model(trainingdata, 2, strings, smoothing=True)

if __name__ == '__main__':
    main()
