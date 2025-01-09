import sys
import os.path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: create_csv.py <dataset_path>")
        sys.exit(1)

    base_path = sys.argv[1]
    separator = ";"
    label = 0

    for dirname, dirnames, filenames in os.walk(base_path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                print ("%s%s%d" % (abs_path, separator, label))
            label = label + 1
